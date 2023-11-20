apt-get update -y
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -y
apt-get install -y python3.9 python3.9-dev python3.9-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
apt-get install -y python3-pip
python3 -m pip install torchdata==0.6.0 torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install h5py==3.8.0 hdf5plugin hydra-core==1.3.2 einops==0.6.0 tqdm numba  pytorch-lightning==1.8.6 wandb==0.14.0 pandas==1.5.3 plotly==5.13.1 opencv-python==4.6.0.66 tabulate==0.9.0 pycocotools==2.0.6 bbox-visualizer==0.1.0 StrEnum==0.4.10
apt-get install -y git vim tmux htop
apt-get install -y ffmpeg libsm6 libxext6
python3  -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
