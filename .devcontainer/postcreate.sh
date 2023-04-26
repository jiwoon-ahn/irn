
sudo apt update && sudo apt -y install libgl1-mesa-glx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r requirements-bad-dependency.txt