export WANDB_API_KEY=a9d82786a985fec4744059efb2034525098a31e9
git config --global user.name Minsu Kang
git config --global user.email powerful@postech.ac.kr
apt update && apt -y install libgl1-mesa-glx
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r /workspaces/irn/.devcontainer/requirements.txt
pip install -r /workspaces/irn/.devcontainer/requirements-bad-dependency.txt