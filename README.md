# 🧠 EVOLVE : Event-Guided Deformable Feature Transfer and Dual-Memory Refinement for Low-Light Video Object Segmentation (ICCV 2025)

<img src="https://github.com/whdgusdl48/EVOLVE/blob/main/assets/Main.png" />
</div>

## News
- We will release code. Coming Soon!!!!

## 📌 Key Features

- 🎯 **Event-guided Deformable Feature Transfer Module**  

- 🔁 **Dual-Memory Object Transformer**  

- 🧩 **Memory Refinement Module**  
---

## Data preparation & Installation
See [Datasets](datasets/Readme.md) & [Installation](INSTALL.md)

## Training Command

We trained with four A100 GPUs, which took around 30 hours.

```
OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 evolve/train.py exp_id=[some unique id] model=[small/base] data=[base/with-mose/mega]
```

- Change `nproc_per_node` to change the number of GPUs.
- Prepend `CUDA_VISIBLE_DEVICES=...` if you want to use specific GPUs.
- Change `master_port` if you encounter port collision.
- `exp_id` is a unique experiment identifier that does not affect how the training is done.
- Models and visualizations will be saved in `./output/`.
- For pre-training only, specify `main_training.enabled=False`.
- For main training only, specify `pre_training.enabled=False`.
- To load a pre-trained model, e.g., to continue main training from the final model from pre-training, specify `weights=[path to the model]`.



## Qualititative Result
- LLE-VOS Dataset
<img src="https://github.com/whdgusdl48/EVOLVE/blob/main/assets/Result_LLEVOS.png" />

- LLE-DAVIS Dataset
<img src="https://github.com/whdgusdl48/EVOLVE/blob/main/assets/Result_LLEDAVIS.png" />

## Citation
```BibTeX
@InProceedings{Baek_2025_ICCV,
    author    = {Baek, Jong-Hyeon and Oh, Jiwon and Koh, Yeong Jun},
    title     = {EVOLVE: Event-Guided Deformable Feature Transfer and Dual-Memory Refinement for Low-Light Video Object Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {11273-11282}
}
```


