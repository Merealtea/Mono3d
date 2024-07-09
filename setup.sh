conda create -n mono3d python=3.8 -y
conda activate mono3d

# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c nvidia cuda==11.6 
conda install -c nvidia cuda-nvcc==11.6.124 
conda install nvidia/label/cuda-11.6.1::libcusolver-dev 
pip install -r requirements.txt