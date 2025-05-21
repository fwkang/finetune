
```bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.11 python3.11-venv python3.11-distutils -y

curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
sudo update-alternatives --config python3

sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.11 2
sudo update-alternatives --config pip3

ln -s /usr/bin/python3 /usr/bin/python

```

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 然后再安装 requirements.txt 中的其他内容
pip install -r requirements.txt
```
