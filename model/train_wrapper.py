import logging
from omegaconf import DictConfig
import numpy as np
import torch
from typing import Dict
from cutie.model.feature_attention import CrossAttentionModule
from einops.layers.torch import Rearrange
from cutie.model.cutie import CUTIE

log = logging.getLogger()


class CutieTrainWrapper(CUTIE):
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__(cfg, single_object=(stage_cfg.num_objects == 1))

        self.sensory_dim = cfg.model.sensory_dim
        self.seq_length = stage_cfg.seq_length
        self.num_ref_frames = stage_cfg.num_ref_frames
        self.deep_update_prob = stage_cfg.deep_update_prob
        self.use_amp = stage_cfg.amp
        self.move_t_out_of_batch = Rearrange('(b t) c h w -> b t c h w', t=self.seq_length)
        self.move_t_from_batch_to_volume = Rearrange('(b t) c h w -> b c t h w', t=self.seq_length)

    def forward(self, data: Dict):
        out = {}
        frames = data['rgb']
        events = data['events']
        first_frame_gt = data['first_frame_gt'].float()
        b, seq_length = frames.shape[:2]

        # 여기서 배치 내에 있는 오브젝트 수들을 계산
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        # 배치 내에 있는 오브젝트 수들 중 최댓값 계산
        max_num_objects = max(num_filled_objects)
        first_frame_gt = first_frame_gt[:, :, :max_num_objects]
        selector = data['selector'][:, :max_num_objects].unsqueeze(2).unsqueeze(2)

        num_objects = first_frame_gt.shape[2]
        out['num_filled_objects'] = num_filled_objects

        def get_ms_feat_ti(ti):
            return [f[:, ti] for f in ms_feat]

        def get_ms_event_ti(ti):
            return [f[:, ti] for f in ms_event]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            frames_flat = frames.view(b * seq_length, *frames.shape[2:])
            events_flat = events.view(b * seq_length, *events.shape[2:])
            ms_feat, ms_event, pix_feat, event_feat = self.encode_image(frames_flat, events_flat)

            with torch.cuda.amp.autocast(enabled=False):
                keys, shrinkages, selections, offset = self.transform_key(ms_feat[0].float(), ms_event[0].float())
                e_keys, e_shrinkages, e_selections = self.transform_event_key(ms_event[0].float())

            # ms_feat: tuples of (B*T)*C*H*W -> B*T*C*H*W
            # keys/shrinkages/selections: (B*T)*C*H*W -> B*C*T*H*W
            
            ms_feat[0] = ms_feat[0] + ms_event[0]
            ms_feat[1] = ms_feat[1] + ms_event[1]
            ms_feat[2] = ms_feat[2] + ms_event[2]


            h, w = keys.shape[-2:]
            keys = self.move_t_from_batch_to_volume(keys)
            shrinkages = self.move_t_from_batch_to_volume(shrinkages)
            selections = self.move_t_from_batch_to_volume(selections)

            e_keys = self.move_t_from_batch_to_volume(e_keys)
            e_shrinkages = self.move_t_from_batch_to_volume(e_shrinkages)
            e_selections = self.move_t_from_batch_to_volume(e_selections)

            offset = self.move_t_from_batch_to_volume(offset)
            ms_feat = [self.move_t_out_of_batch(f) for f in ms_feat]
            ms_event = [self.move_t_out_of_batch(f) for f in ms_event]
            pix_feat = self.move_t_out_of_batch(pix_feat)
            event_feat = self.move_t_out_of_batch(event_feat)
            # zero-init sensory
            sensory = torch.zeros((b, num_objects, self.sensory_dim, h, w), device=frames.device)
            msk_val, sensory, obj_val, _, event_val = self.encode_mask(frames[:, 0], pix_feat[:, 0], event_feat[:, 0],
                                                                       sensory,
                                                                       first_frame_gt[:, 0])
            masks = first_frame_gt[:, 0]

            # add the time dimension
            msk_values = msk_val.unsqueeze(3)  # B*num_objects*C*T*H*W
            obj_values = obj_val.unsqueeze(
                2) if obj_val is not None else None  # B*num_objects*T*Q*C
            e_obj_values = event_val.unsqueeze(
                2) if obj_val is not None else None  # B*num_objects*T*Q*C

            for ti in range(1, seq_length):
                if ti <= self.num_ref_frames:
                    ref_msk_values = msk_values
                    ref_keys = keys[:, :, :ti]
                    ref_shrinkages = shrinkages[:, :, :ti] if shrinkages is not None else None
                    ref_e_keys = e_keys[:, :, :ti]
                    ref_e_shrinkages = e_shrinkages[:, :, :ti] if e_shrinkages is not None else None
                    ref_msk_values = self.mask_offset_conv(ref_msk_values, offset[:, :, :ti].unsqueeze(1))
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would
                    # need broadcd casting in gather which we don't have

                    ridx = [torch.randperm(ti)[:self.num_ref_frames].sort()[0] for _ in range(b)]
                    #first_frame_gt is first
                    for i in range(len(ridx)):
                        ridx[i][0] = 0
                    ref_msk_values = torch.stack(
                        [msk_values[bi, :, :, ridx[bi]] for bi in range(b)], 0)
                    ref_keys = torch.stack([keys[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_shrinkages = torch.stack([shrinkages[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_e_keys = torch.stack([e_keys[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_e_shrinkages = torch.stack([e_shrinkages[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_offset = torch.stack([offset[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_msk_values = self.mask_offset_conv(ref_msk_values, ref_offset.unsqueeze(1))
                # Segment frame ti
                readout, e_readout, aux_input = self.read_memory(keys[:, :, ti], selections[:, :, ti],
                                                      e_keys[:, :, ti], e_selections[:, :, ti],
                                                      ref_keys, ref_shrinkages, ref_e_keys, ref_e_shrinkages,
                                                      ref_msk_values, obj_values, e_obj_values,
                                                      pix_feat[:, ti], event_feat[:, ti], sensory, masks, selector)
                aux_output = self.compute_aux(pix_feat[:, ti], aux_input, selector)

                sensory, logits, masks = self.segment(get_ms_feat_ti(ti),
                                                      get_ms_event_ti(ti),
                                                      readout,
                                                      e_readout,
                                                      sensory,
                                                      selector=selector)
                # remove background
                masks = masks[:, 1:]

                # No need to encode the last frame
                if ti < (self.seq_length - 1):
                    deep_update = np.random.rand() < self.deep_update_prob
                    msk_val, sensory, obj_val, _, e_obj_val = self.encode_mask(frames[:, ti],
                                                                               pix_feat[:, ti],
                                                                               event_feat[:, ti],
                                                                               sensory,
                                                                               masks,
                                                                               deep_update=deep_update)
                    msk_values = torch.cat([msk_values, msk_val.unsqueeze(3)], 3)
                    obj_values = torch.cat([obj_values, obj_val.unsqueeze(2)],
                                           2) if obj_val is not None else None
                    e_obj_values = torch.cat([e_obj_values, e_obj_val.unsqueeze(2)],
                                             2) if e_obj_val is not None else None

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits
                out[f'aux_{ti}'] = aux_output

        return out
