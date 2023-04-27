FROM mcr.microsoft.com/devcontainers/python:0-3.10

RUN apt update && apt -y install libgl1-mesa-glx

ADD . .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
RUN pip install -r requirements-bad-dependency.txt