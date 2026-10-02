FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# The webui and TensorRT worker deliberately use separate Python environments.
RUN apt update && apt install -y python3.12 python3.12-venv python3-pip git build-essential openmpi-bin libopenmpi-dev && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/opt/webui/bin:$PATH

# Build from the fork checkout: docker build -f docker/TensorRT-LLM/Dockerfile .
COPY . /app/textgen
WORKDIR /app/textgen
RUN python3.12 -m venv /opt/webui && \
    python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128 && \
    python -m pip install -r requirements/full/requirements.txt && \
    python scripts/setup_tensorrt.py --python /usr/bin/python3.12 --runtime-dir /opt/tensorrt_runtime

EXPOSE 7860 5000
CMD ["python", "server.py", "--api", "--listen", "--tensorrt-llm-python", "/opt/tensorrt_runtime/bin/python"]
