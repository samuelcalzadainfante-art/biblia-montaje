"""
handler.py — RunPod Serverless · Montaje de Vídeo Bíblico
"""

import os, json, base64, logging, subprocess, shutil, time, tempfile, sys
import requests
import runpod

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)
log = logging.getLogger("montaje-worker")

IMAGE_WIDTH  = 1536
IMAGE_HEIGHT = 1024
VIDEO_FPS    = 30
VIDEO_CRF    = 18

MUSIC_CANDIDATES = [
    "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Vanishing.mp3",
    "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Ossuary%205%20-%20Rest.mp3",
    "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Impact%20Moderato.mp3",
]

YELLOW = "&H0000FFFF"
WHITE  = "&H00FFFFFF"

ASS_HEADER = (
    "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    f"Style: Default,Liberation Sans,65,{WHITE},{WHITE},&H00000000,&H00000000,"
    "1,0,0,0,100,100,0,0,1,4,0,2,10,10,60,1\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)

_whisper_model = None

def _log_gpu_info():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        log.info(f"  CUDA disponible: {cuda_ok}")
        if cuda_ok:
            log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram_total = props.total_memory / 1024**3
            vram_used  = torch.cuda.memory_allocated(0) / 1024**3
            log.info(f"  VRAM total: {vram_total:.1f} GB — usada: {vram_used:.1f} GB")
        else:
            log.warning("  CUDA no disponible — Whisper correrá en CPU")
    except Exception as e:
        log.warning(f"  No se pudo obtener info GPU: {e}")

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _log_gpu_info()
        try:
            import torch, whisper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"  Cargando openai-whisper large en {device}...")
            _whisper_model = ("openai", whisper.load_model("large", device=device))
            log.info("  openai-whisper large listo.")
        except Exception as e:
            log.warning(f"  openai-whisper falló al cargar: {e}")
            try:
                from faster_whisper import WhisperModel
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute = "float16" if device == "cuda" else "int8"
                log.info(f"  Cargando faster-whisper large-v3 en {device}...")
                _whisper_model = ("faster", WhisperModel("large-v3", device=device, compute_type=compute))
                log.info("  faster-whisper listo.")
            except Exception as e2:
                log.error(f"  faster-whisper también falló: {e2}")
                _whisper_model = None
    return _whisper_model


def _transcribe(audio_path):
    model_info = _get_whisper()
    if model_info is None:
        raise RuntimeError("Ningún modelo Whisper disponible")
    kind, model = model_info
    words_timing = []

    if kind == "openai":
        import torch
        log.info("  Transcribiendo con openai-whisper...")
        result = model.transcribe(audio_path, language="es", word_timestamps=True,
                                  fp16=torch.cuda.is_available())
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                raw = w.get("word", "").strip()
                if raw:
                    words_timing.append({"word": raw, "start": float(w["start"]), "end": float(w["end"])})
        log.info(f"  openai-whisper devolvió {len(words_timing)} palabras")
        if len(words_timing) == 0:
            raise RuntimeError(
                "openai-whisper transcribió OK pero devolvió 0 palabras "
                "(posible falta de dtw-python o word_timestamps vacío) — "
                "intentando faster-whisper"
            )
    else:  # faster-whisper
        log.info("  Transcribiendo con faster-whisper...")
        segments, _ = model.transcribe(audio_path, language="es", word_timestamps=True)
        for seg in segments:
            for w in seg.words:
                if w.word.strip():
                    words_timing.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
        log.info(f"  faster-whisper devolvió {len(words_timing)} palabras")

    return words_timing


def run_ffmpeg(cmd, desc=""):
    if desc:
        log.info(f"  FFmpeg: {desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error [{desc}]:\n{result.stderr[-800:]}")
    return result


def download_file(url, dest, desc=""):
    log.info(f"  ↓ {desc or os.path.basename(dest)}")
    with requests.get(url, stream=True, timeout=180, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    log.info(f"    ✓ {os.path.getsize(dest)//1024} KB")


def apply_ken_burns(imagen_path, duracion_seg, output_path, escena_num):
    tipos = ["zoom_in", "zoom_out", "pan_right", "pan_left", "zoom_in"]
    tipo  = tipos[escena_num % len(tipos)]
    n_frames  = max(int(duracion_seg * VIDEO_FPS), 1)
    zoom_step = 0.15 / n_frames
    pan_step  = IMAGE_WIDTH * 0.10 / n_frames

    if tipo == "zoom_in":
        zoompan = (f"zoompan=z='min(1.0+on*{zoom_step:.8f},1.15)':d=1"
                   f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                   f":s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:fps={VIDEO_FPS}")
    elif tipo == "zoom_out":
        zoompan = (f"zoompan=z='max(1.15-on*{zoom_step:.8f},1.0)':d=1"
                   f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                   f":s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:fps={VIDEO_FPS}")
    elif tipo == "pan_right":
        zoompan = (f"zoompan=z=1.05:d=1:x='min(on*{pan_step:.6f},iw*0.10)'"
                   f":y='ih/2-(ih/zoom/2)':s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:fps={VIDEO_FPS}")
    else:
        zoompan = (f"zoompan=z=1.05:d=1:x='max(iw*0.10-on*{pan_step:.6f},0)'"
                   f":y='ih/2-(ih/zoom/2)':s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:fps={VIDEO_FPS}")

    run_ffmpeg(["ffmpeg", "-y", "-loop", "1", "-i", imagen_path,
                "-vf", f"{zoompan},trim=duration={duracion_seg},setpts=PTS-STARTPTS",
                "-t", str(duracion_seg), "-c:v", "libx264", "-preset", "fast",
                "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p", output_path],
               f"Ken Burns escena {escena_num:03d}")
    return output_path


def calcular_duraciones(n_escenas, duracion_total_seg):
    FASE1_SEG = 25 * 60
    DUR_FASE1 = 18.0
    n_fase1   = min(n_escenas, int(FASE1_SEG / DUR_FASE1))
    t_fase1   = n_fase1 * DUR_FASE1
    n_fase2   = n_escenas - n_fase1
    dur_fase2 = (duracion_total_seg - t_fase1) / n_fase2 if n_fase2 > 0 else 0.0
    return [DUR_FASE1 if i < n_fase1 else dur_fase2 for i in range(n_escenas)]


def _fmt_ass(seg):
    seg = max(0.0, seg)
    h = int(seg // 3600); m = int((seg % 3600) // 60); s = int(seg % 60)
    cs = min(int(round((seg - int(seg)) * 100)), 99)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _build_karaoke_line(grupo_words, idx_activa):
    partes = []
    for k, w in enumerate(grupo_words):
        if k == idx_activa:
            partes.append(f"{{\\c{YELLOW}&}}{w}{{\\c{WHITE}&}}")
        else:
            partes.append(w)
    return " ".join(partes)


def generar_ass_whisper(words_timing, duracion_total, ass_path):
    os.makedirs(os.path.dirname(ass_path), exist_ok=True)
    words = [{"word": w["word"].strip().upper(), "start": w["start"], "end": w["end"]}
             for w in words_timing if w["word"].strip()]
    events = []
    for i in range(0, len(words), 4):
        grupo = words[i:i+4]
        grupo_txt = [w["word"] for w in grupo]
        for j, wi in enumerate(grupo):
            t_fin = min(wi["end"], duracion_total)
            line  = _build_karaoke_line(grupo_txt, j)
            events.append(f"Dialogue: 0,{_fmt_ass(wi['start'])},{_fmt_ass(t_fin)},Default,,0,0,0,,{line}")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER + "\n".join(events))
    log.info(f"  ✓ ASS Whisper: {len(events)} eventos")
    return ass_path


def generar_ass_guion(escenas, duracion_total, ass_path):
    os.makedirs(os.path.dirname(ass_path), exist_ok=True)
    grupos = []
    for escena in escenas:
        for parrafo in [p.strip() for p in escena["texto_narrador"].split("\n") if p.strip()]:
            palabras = parrafo.upper().split()
            for i in range(0, len(palabras), 4):
                grupos.append(palabras[i:i+4])
    total_palabras  = sum(len(g) for g in grupos)
    seg_por_palabra = duracion_total / max(total_palabras, 1)
    events = []
    t = 0.0
    for grupo in grupos:
        dur_g = len(grupo) * seg_por_palabra
        dur_w = dur_g / max(len(grupo), 1)
        for j, w in enumerate(grupo):
            t_ini = t + j * dur_w
            t_fin = min(t_ini + dur_w, duracion_total)
            events.append(f"Dialogue: 0,{_fmt_ass(t_ini)},{_fmt_ass(t_fin)},Default,,0,0,0,,{_build_karaoke_line(grupo, j)}")
        t = min(t + dur_g, duracion_total)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER + "\n".join(events))
    log.info(f"  ✓ ASS guión: {len(events)} eventos")
    return ass_path


def upload_to_drive(file_path, folder_id, sa_json_raw, rclone_token=""):
    filename = os.path.basename(file_path)
    cfg_content = f"""[gdrive]
type = drive
scope = drive
token = {rclone_token}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as cfg_file:
        cfg_file.write(cfg_content)
        cfg_path = cfg_file.name
    try:
        cmd = ["rclone", "copy", "--config", cfg_path, file_path, f"gdrive:{folder_id}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rclone error: {result.stderr}")
        ls_cmd = ["rclone", "lsjson", "--config", cfg_path, f"gdrive:{folder_id}", "--files-only"]
        ls_result = subprocess.run(ls_cmd, capture_output=True, text=True)
        files = json.loads(ls_result.stdout) if ls_result.stdout.strip() else []
        file_id = next((f["ID"] for f in files if f["Name"] == filename), None)
        url = f"https://drive.google.com/file/d/{file_id}/view" if file_id else ""
        log.info(f"  ✓ Drive: {url}")
        return file_id, url
    finally:
        os.unlink(cfg_path)


def conseguir_musica(workdir, musica_url=""):
    dest = os.path.join(workdir, "musica_ambiente.mp3")
    for url in ([musica_url] if musica_url else []) + MUSIC_CANDIDATES:
        try:
            download_file(url, dest, "música ambiente")
            if os.path.getsize(dest) > 100_000:
                return dest
        except Exception as e:
            log.warning(f"  Música fallida: {e}")
    return None


def handler(job):
    t0  = time.time()
    inp = job.get("input", {})
    slug          = inp.get("slug", "video")
    audio_url     = inp.get("audio_url", "")
    imagenes_urls = inp.get("imagenes_urls", [])
    escenas       = inp.get("escenas", [])
    duracion_seg  = float(inp.get("duracion_seg", 0))
    musica_url    = inp.get("musica_url", "")
    folder_id     = inp.get("drive_folder_id", "")
    sa_json       = inp.get("gdrive_sa_json", "")
    rclone_token  = inp.get("rclone_token", "")

    if not audio_url:     return {"error": "Falta 'audio_url'", "success": False}
    if not imagenes_urls: return {"error": "Falta 'imagenes_urls'", "success": False}
    if not duracion_seg:  return {"error": "Falta 'duracion_seg'", "success": False}

    workdir   = f"/tmp/{slug}"
    clips_dir = f"{workdir}/clips"
    subs_dir  = f"{workdir}/subtitulos"
    imgs_dir  = f"{workdir}/imagenes"
    for d in [workdir, clips_dir, subs_dir, imgs_dir]:
        os.makedirs(d, exist_ok=True)

    words_count = 0
    try:
        log.info(f"MONTAJE: {slug} — {len(imagenes_urls)} escenas — {duracion_seg/60:.1f} min")

        audio_path = f"{workdir}/audio.mp3"
        download_file(audio_url, audio_path, "audio MP3")

        imagen_paths = []
        for i, url in enumerate(imagenes_urls, 1):
            dest = f"{imgs_dir}/escena_{i:03d}.png"
            try:
                download_file(url, dest, f"escena {i:03d}")
                imagen_paths.append(dest)
            except Exception as e:
                log.warning(f"  ⚠ escena {i:03d}: {e}")
                imagen_paths.append(None)

        duraciones = calcular_duraciones(len(imagen_paths), duracion_seg)
        clips = []
        for i, (img, dur) in enumerate(zip(imagen_paths, duraciones), 1):
            out = f"{clips_dir}/kb_{i:03d}.mp4"
            if os.path.exists(out):
                clips.append(out); continue
            if not img or not os.path.exists(img):
                run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi",
                            "-i", f"color=c=black:s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:r={VIDEO_FPS}",
                            "-t", str(dur), "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                           f"placeholder {i:03d}")
            else:
                apply_ken_burns(img, dur, out, i)
            clips.append(out)

        lista_txt = f"{workdir}/lista_clips.txt"
        with open(lista_txt, "w") as f:
            for c in clips:
                if c and os.path.exists(c): f.write(f"file '{c}'\n")
        video_sin_audio = f"{workdir}/video_sin_audio.mp4"
        run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lista_txt,
                    "-c:v", "libx264", "-preset", "fast", "-crf", str(VIDEO_CRF),
                    "-pix_fmt", "yuv420p", video_sin_audio], "concat clips")

        words_timing = []
        try:
            log.info("  Iniciando transcripción Whisper...")
            words_timing = _transcribe(audio_path)
            words_count = len(words_timing)
        except Exception as e:
            log.error(f"  ✗ Whisper falló completamente: {e}")
            # segundo intento con faster-whisper directo
            try:
                log.info("  Intentando faster-whisper directamente...")
                from faster_whisper import WhisperModel
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute = "float16" if device == "cuda" else "int8"
                fw_model = WhisperModel("large-v3", device=device, compute_type=compute)
                segments, _ = fw_model.transcribe(audio_path, language="es", word_timestamps=True)
                for seg in segments:
                    for w in seg.words:
                        if w.word.strip():
                            words_timing.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
                words_count = len(words_timing)
                log.info(f"  faster-whisper directo: {words_count} palabras")
            except Exception as e2:
                log.error(f"  ✗ faster-whisper directo también falló: {e2}")

        ass_path = f"{subs_dir}/subtitulos.ass"
        if words_timing:
            generar_ass_whisper(words_timing, duracion_seg, ass_path)
        elif escenas:
            log.warning("  Usando fallback subtítulos por guión")
            generar_ass_guion(escenas, duracion_seg, ass_path)
        else:
            ass_path = None

        musica_path = conseguir_musica(workdir, musica_url)

        video_final    = f"{workdir}/{slug}_FINAL.mp4"
        ass_for_ffmpeg = ass_path.replace("\\", "/") if ass_path else None

        if ass_for_ffmpeg and musica_path:
            filter_complex = (
                "[1:a]volume=1.0[narr];[2:a]volume=0.12[amb];"
                "[narr][amb]amix=inputs=2:duration=first[aout];"
                f"[0:v]ass='{ass_for_ffmpeg}'[vout]"
            )
            cmd = ["ffmpeg", "-y", "-i", video_sin_audio, "-i", audio_path,
                   "-stream_loop", "-1", "-i", musica_path,
                   "-filter_complex", filter_complex,
                   "-map", "[vout]", "-map", "[aout]",
                   "-c:a", "aac", "-b:a", "192k", "-c:v", "libx264",
                   "-preset", "medium", "-crf", str(VIDEO_CRF),
                   "-pix_fmt", "yuv420p", "-t", str(duracion_seg), video_final]
        elif ass_for_ffmpeg:
            cmd = ["ffmpeg", "-y", "-i", video_sin_audio, "-i", audio_path,
                   "-vf", f"ass='{ass_for_ffmpeg}'", "-map", "0:v", "-map", "1:a",
                   "-c:a", "aac", "-b:a", "192k", "-c:v", "libx264",
                   "-preset", "medium", "-crf", str(VIDEO_CRF),
                   "-pix_fmt", "yuv420p", "-t", str(duracion_seg), video_final]
        else:
            cmd = ["ffmpeg", "-y", "-i", video_sin_audio, "-i", audio_path,
                   "-map", "0:v", "-map", "1:a",
                   "-c:a", "aac", "-b:a", "192k", "-c:v", "libx264",
                   "-preset", "medium", "-crf", str(VIDEO_CRF),
                   "-pix_fmt", "yuv420p", "-t", str(duracion_seg), video_final]

        run_ffmpeg(cmd, "montaje final")

        probe = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                                video_final], capture_output=True, text=True)
        dur_final = float(probe.stdout.strip())
        size_mb   = os.path.getsize(video_final) / 1_048_576
        log.info(f"  ✓ VÍDEO: {dur_final/60:.1f} min — {size_mb:.0f} MB")

        drive_file_id = drive_url = None
        if folder_id and rclone_token:
            drive_file_id, drive_url = upload_to_drive(video_final, folder_id, sa_json, rclone_token)

        elapsed = (time.time() - t0) / 60
        return {"success": True, "drive_file_id": drive_file_id, "drive_url": drive_url,
                "duracion_final_seg": round(dur_final, 1), "size_mb": round(size_mb, 1),
                "elapsed_min": round(elapsed, 1), "whisper_words": words_count}

    except Exception as exc:
        log.exception(f"ERROR: {exc}")
        return {"success": False, "error": str(exc)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
