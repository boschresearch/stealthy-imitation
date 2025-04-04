# !/bin/bash

# Create conda environment
conda create --name si_mujoco python=3.8.13 -y

# Rename activate to avoid potential conflicts
sudo mv /opt/conda/bin/activate /opt/conda/bin/activate_no

conda install --name si_mujoco -y \
-c pytorch \
-c defaults \
_libgcc_mutex=0.1=main \
_openmp_mutex=5.1=1_gnu \
blas=1.0=mkl \
brotlipy=0.7.0=py38h27cfd23_1003 \
bzip2=1.0.8=h7b6447c_0 \
cffi=1.15.0=py38hd667e15_1 \
charset-normalizer=2.0.4=pyhd3eb1b0_0 \
cryptography=37.0.1=py38h9ce1e76_0 \
cudatoolkit=11.3.1=h2bc3f7f_2 \
ffmpeg=4.3=hf484d3e_0 \
freetype=2.11.0=h70c0345_0 \
giflib=5.2.1=h7b6447c_0 \
gmp=6.2.1=h295c915_3 \
gnutls=3.6.15=he1e5248_0 \
idna=3.3=pyhd3eb1b0_0 \
intel-openmp=2021.4.0=h06a4308_3561 \
jpeg=9e=h7f8727e_0 \
lame=3.100=h7b6447c_0 \
lcms2=2.12=h3be6417_0 \
ld_impl_linux-64=2.38=h1181459_1 \
libffi=3.3=he6710b0_2 \
libgcc-ng=11.2.0=h1234567_1 \
libgomp=11.2.0=h1234567_1 \
libiconv=1.16=h7f8727e_2 \
libidn2=2.3.2=h7f8727e_0 \
libpng=1.6.37=hbc83047_0 \
libstdcxx-ng=11.2.0=h1234567_1 \
libtasn1=4.16.0=h27cfd23_0 \
libtiff=4.2.0=h2818925_1 \
libunistring=0.9.10=h27cfd23_0 \
libwebp=1.2.2=h55f646e_0 \
libwebp-base=1.2.2=h7f8727e_0 \
lz4-c=1.9.3=h295c915_1 \
mkl=2021.4.0=h06a4308_640 \
mkl-service=2.4.0=py38h7f8727e_0 \
mkl_fft=1.3.1=py38hd3c417c_0 \
mkl_random=1.2.2=py38h51133e4_0 \
ncurses=6.3=h7f8727e_2 \
nettle=3.7.3=hbbd107a_1 \
numpy=1.22.3=py38he7a7128_0 \
numpy-base=1.22.3=py38hf524024_0 \
openh264=2.1.1=h4ff587b_0 \
openssl=1.1.1o=h7f8727e_0 \
pillow=9.0.1=py38h22f2fdc_0 \
pip=21.2.4=py38h06a4308_0 \
pycparser=2.21=pyhd3eb1b0_0 \
pyopenssl=22.0.0=pyhd3eb1b0_0 \
pysocks=1.7.1=py38h06a4308_0 \
python=3.8.13=h12debd9_0 \
pytorch=1.12.0=py3.8_cuda11.3_cudnn8.3.2_0 \
pytorch-mutex=1.0=cuda \
readline=8.1.2=h7f8727e_1 \
requests=2.27.1=pyhd3eb1b0_0 \
setuptools=61.2.0=py38h06a4308_0 \
six=1.16.0=pyhd3eb1b0_1 \
sqlite=3.38.5=hc218d9a_0 \
tk=8.6.12=h1ccaba5_0 \
torchaudio=0.12.0=py38_cu113 \
torchvision=0.13.0=py38_cu113 \
typing_extensions=4.1.1=pyh06a4308_0 \
urllib3=1.26.9=py38h06a4308_0 \
wheel=0.37.1=pyhd3eb1b0_0 \
xz=5.2.5=h7f8727e_1 \
zlib=1.2.12=h7f8727e_2 \
zstd=1.5.2=ha4553b6_0

# Update apt packages
sudo apt update
sudo apt install libgl1-mesa-glx libosmesa6-dev libglib2.0-0 libsm6 libxext6 libxrender-dev swig curl git vim gcc g++ make wget locales dnsutils zip unzip cmake build-essential -y
sudo apt clean
sudo rm -rf /var/cache/apt/*



# Setup Mujoco
mkdir -p /home/myuser/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
tar -xf mujoco.tar.gz -C /home/myuser/.mujoco
rm mujoco.tar.gz
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/myuser/.mujoco/mjpro210/bin:/home/myuser/.mujoco/mujoco210/bin" >> /home/myuser/.bashrc

export LD_LIBRARY_PATH=/home/myuser/.mujoco/mjpro210/bin:/home/myuser/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

# Define the path to the pip of the si_mujoco environment
PIP_PATH="/home/myuser/.conda/envs/si_mujoco/bin/pip"

# Use the specific pip to install python packages
$PIP_PATH install di-engine==0.4.9 mujoco==2.3.7 imageio==2.31.1 mujoco-py==2.1.2.14 cython==0.29.36 patchelf==0.17.2.1 numba==0.58.1 statsmodels==0.14.0
$PIP_PATH install --no-cache-dir numpy==1.22.3
$PIP_PATH install --no-cache-dir -U "gym[mujoco,mujoco_py]==0.25.1"
$PIP_PATH install gymnasium[mujoco]==0.29.1 ipdb==0.13.13 torchvision==0.18.0 protobuf==4.25

sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
