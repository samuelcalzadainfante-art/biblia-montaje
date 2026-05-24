FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        ffmpeg \
        libass9 \
        fonts-liberation \
        fontconfig \
        curl \
        unzip \
    && ln -sf python3.10 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        torch==2.1.0 \
        torchvision==0.16.0 \
        torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
        runpod==1.9.0 \
        numpy \
        openai-whisper \
        faster-whisper \
        requests \
        pillow \
        tqdm \
        google-api-python-client==2.111.0 \
        google-auth==2.25.2 \
        google-auth-httplib2==0.2.0 \
        google-auth-oauthlib==1.2.0

RUN curl -L "https://downloads.rclone.org/rclone-current-linux-amd64.zip" \
        -o /tmp/rclone.zip \
    && unzip /tmp/rclone.zip -d /tmp/rclone_dir \
    && mv /tmp/rclone_dir/rclone-*/rclone /usr/local/bin/ \
    && rm -rf /tmp/rclone_dir /tmp/rclone.zip

COPY handler.py .
CMD ["python", "handler.py"]
