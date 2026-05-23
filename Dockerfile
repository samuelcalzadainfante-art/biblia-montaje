# ── Montaje RunPod Worker ──────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libass9 \
        fonts-liberation \
        fontconfig \
        curl \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        runpod==1.7.3 \
        openai-whisper \
        requests \
        pillow \
        tqdm \
        google-api-python-client==2.111.0 \
        google-auth==2.25.2 \
        google-auth-httplib2==0.2.0 \
        google-auth-oauthlib==1.2.0

RUN python -c "import whisper; whisper.load_model('large'); print('Whisper large OK')"

COPY handler.py .
ENV PYTHONUNBUFFERED=1
CMD ["python", "handler.py"]
