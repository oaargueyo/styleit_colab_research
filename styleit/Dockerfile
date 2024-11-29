FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/home/appuser/.local/bin:${PATH}" \
    FORCE_CUDA="0" \
    TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-distutils python3.8-dev \
    python3-opencv ca-certificates git wget sudo ninja-build build-essential && \
    ln -svf /usr/bin/python3.8 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py --user && rm get-pip.py

# Python dependencies
RUN pip install --no-cache-dir --user \
    tensorboard cmake onnx \
    torch==1.10+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    fastapi uvicorn pyimgur google-search-results requests psycopg2-binary python-multipart

# Install fvcore and Detectron2
RUN pip install --no-cache-dir --user 'git+https://github.com/facebookresearch/fvcore' && \
    git clone https://github.com/facebookresearch/detectron2 detectron2_repo && \
    pip install --no-cache-dir --user -e detectron2_repo

# Set model cache directory
ENV FVCORE_CACHE="/tmp"

# Application setup
WORKDIR /home/appuser/app
USER appuser
RUN mkdir -p /home/appuser/app/uploads && sudo chmod -R 777 /home/appuser/app/uploads

# Copy application files and requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --requirement /app/requirements.txt
COPY ./app.py /home/appuser/app/
COPY ./styleit_cv.py /home/appuser/app/

# Expose port and set CMD
EXPOSE 8888
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--reload", "--port", "8888"]