import os
import cv2
from os import path as osp
import glob
import numpy as np
import shutil
from event_util import generate_voxel_grid
from tqdm import tqdm

event_bins = [5]
image_path = "Image Path"
for event_bin in event_bins:

    event_path = "Event_path"
    target_event_path = "target_event_path" + str(event_bin)
    target_eventvis_path = "target_eventvis_path" + str(event_bin)

    separate_pol = False

    for dir in tqdm(os.listdir(event_path)):
        dir_path = osp.join(event_path, dir)
        event_list = sorted(glob.glob(osp.join(dir_path, '*.npz')))

        target_dir_path = osp.join(target_event_path, dir)
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
 
        target_eventvis_dir_path = osp.join(target_eventvis_path, dir)
        if not os.path.exists(target_eventvis_dir_path):
            os.makedirs(target_eventvis_dir_path)

        image = cv2.imread(osp.join(image_path, dir, '00001.jpg'))
        shape = image.shape[0:2]

        for event_file_name in event_list:
            npz_file = np.load(event_file_name)
            x = npz_file['x']
            y = npz_file['y']
            t = npz_file['t']
            p = npz_file['p']

            # 예: (N,) 짜리 1D 배열 4개를 (N,4) 2D 배열로 합치기
            event_concat = np.column_stack((x, y, t, p))
            event_voxel = generate_voxel_grid(event_concat, shape, event_bin, separate_pol)
            
            eventvoxel_filename = osp.join(target_dir_path, "{}.npy".format(os.path.basename(event_file_name).split(".")[0]))
            np.save(eventvoxel_filename, event_voxel)

            voxel_image_path = osp.join(
                target_eventvis_dir_path, "{}.jpg".format(os.path.basename(event_file_name).split(".")[0])
            )

    print(f"{event_bin} voxel complete")
