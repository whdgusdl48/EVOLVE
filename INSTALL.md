## Installation

### Requirements
- Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- OpenCV is optional but needed by demo and visualization
- Panopticapi: `pip install git+https://github.com/cocodataset/panopticapi.git`
- `pip install -r requirements.txt`

### Example conda environment setup
```bash
conda create --name cavis python=3.8 -y
conda activate cavis
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U opencv-python

# download this repository and unzip
cd Evolve
pip install -e .
```
