# ── Montaje RunPod Worker ──────────────────────────────────────────────────────
# Stack : FFmpeg + Whisper large (GPU) + Ken Burns + Subtítulos Karaoke ASS
# Imagen: kuekuatsu17/biblia-montaje
# GPU   : CUDA 12.1 — mínimo RTX 3090 / A40 (Whisper large ~3 GB VRAM)
# Input : {slug, audio_url, imagenes_urls, escenas, duracion_seg, ...}
# Output: MP4 final subido a Google Drive vía Service Account
# ──────────────────────────────────────────────────────────────────────────────

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# ── Sistema: FFmpeg (con libass, libx264), fuentes, curl ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libass9 \
        fonts-liberation \
        fontconfig \
        curl \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# ── Python: Whisper + Google Drive API + RunPod + utilidades ──────────────────
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

# ── Pre-cachear modelo Whisper large (evita descarga de 3 GB en cold start) ───
# El modelo queda en /root/.cache/whisper/large.pt dentro de la imagen
RUN python -c "import whisper; whisper.load_model('large'); print('Whisper large OK')"

# ── Handler ───────────────────────────────────────────────────────────────────
COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "handler.py"]
