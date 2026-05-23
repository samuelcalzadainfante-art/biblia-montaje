"""
handler.py — RunPod Serverless · Montaje de Vídeo Bíblico
==========================================================
Stack : FFmpeg + Whisper large + Ken Burns + Subtítulos Karaoke ASS

Input JSON:
  slug            : str   — identificador del vídeo (usado como nombre de archivo)
  audio_url       : str   — URL descargable del MP3 de narración
  imagenes_urls   : list  — URLs PNG ordenadas [escena_001, escena_002, ...]
  escenas         : list  — [{numero, texto_narrador, ...}]  ← fallback subtítulos
  duracion_seg    : float — duración exacta del audio en segundos
  musica_url      : str?  — URL MP3 música ambiente (opcional; si falta usa Kevin MacLeod)
  drive_folder_id : str   — ID de la carpeta Drive donde subir el MP4 final
  gdrive_sa_json  : str   — Service Account JSON (base64 o JSON crudo como str)

Output JSON:
  success             : bool
  drive_file_id       : str
  drive_url           : str
  duracion_final_seg  : float
  size_mb             : float
  elapsed_min         : float
  whisper_words       : int   — 0 si Whisper falló y se usó el guión
"""

import os, json, base64, logging, subprocess, shutil, time
import torch
import requests
import runpod
import whisper

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("montaje-worker")

# ── Constantes de vídeo (igual que 05_montaje.py) ────────────────────────────
IMAGE_WIDTH  = 1536
IMAGE_HEIGHT = 1024
VIDEO_FPS    = 30
VIDEO_CRF    = 18

# Fallback música si no se pasa musica_url
MUSIC_CANDIDATES = [
    "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Vanishing.mp3",
    "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Ossuary%205%20-%20Rest.mp3",
    "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Impact%20Moderato.mp3",
]

# ASS colors (&HAABBGGRR)
YELLOW = "&H0000FFFF"
WHITE  = "&H00FFFFFF"

ASS_HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1920\n"
    "PlayResY: 1080\n"
    "WrapStyle: 0\n\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    # Liberation Sans es metric-compatible con Arial y está disponible en Linux
    f"Style: Default,Liberation Sans,65,{WHITE},{WHITE},&H00000000,&H00000000,"
    "1,0,0,0,100,100,0,0,1,4,0,2,10,10,60,1\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)

# ── Cargar Whisper large al arrancar (se baka en la imagen Docker) ────────────
log.info("Cargando Whisper large...")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large", device=_DEVICE)
log.info(f"Whisper large listo en {_DEVICE}.")


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def run_ffmpeg(cmd, desc=""):
    if desc:
        log.info(f"  FFmpeg: {desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error [{desc}]:\n{result.stderr[-800:]}")
    return result


def download_file(url, dest, desc=""):
    log.info(f"  ↓ {desc or os.path.basename(dest)}")
    with requests.get(url, stream=True, timeout=180,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    size_kb = os.path.getsize(dest) // 1024
    log.info(f"    ✓ {size_kb} KB")


# ══════════════════════════════════════════════════════════════════════════════
# KEN BURNS — idéntico a 05_montaje.py
# ══════════════════════════════════════════════════════════════════════════════

def apply_ken_burns(imagen_path, duracion_seg, output_path, escena_num):
    tipos = ["zoom_in", "zoom_out", "pan_right", "pan_left", "zoom_in"]
    tipo  = tipos[escena_num % len(tipos)]

    n_frames  = max(int(duracion_seg * VIDEO_FPS), 1)
    zoom_step = 0.15 / n_frames
    pan_step  = IMAGE_WIDTH * 0.10 / n_frames

    if tipo == "zoom_in":
        zoompan = (
            f"zoompan=z='min(1.0+on*{zoom_step:.8f},1.15)':d=1"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:fps={VIDEO_FPS}"
        )
    elif tipo == "zoom_out":
        zoompan = (
            f"zoompan=z='max(1.15-on*{zoom_step:.8f},1.0)':d=1"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:fps={VIDEO_FPS}"
        )
    elif tipo == "pan_right":
        zoompan = (
            f"zoompan=z=1.05:d=1"
            f":x='min(on*{pan_step:.6f},iw*0.10)':y='ih/2-(ih/zoom/2)'"
            f":s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:fps={VIDEO_FPS}"
        )
    else:  # pan_left
        zoompan = (
            f"zoompan=z=1.05:d=1"
            f":x='max(iw*0.10-on*{pan_step:.6f},0)':y='ih/2-(ih/zoom/2)'"
            f":s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:fps={VIDEO_FPS}"
        )

    run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", imagen_path,
        "-vf", f"{zoompan},trim=duration={duracion_seg},setpts=PTS-STARTPTS",
        "-t", str(duracion_seg),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(VIDEO_CRF),
        "-pix_fmt", "yuv420p",
        output_path
    ], f"Ken Burns escena {escena_num:03d}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# DISTRIBUCIÓN DE DURACIONES — idéntico a 05_montaje.py
# ══════════════════════════════════════════════════════════════════════════════

def calcular_duraciones(n_escenas, duracion_total_seg):
    FASE1_SEG = 25 * 60
    DUR_FASE1 = 18.0
    n_fase1   = min(n_escenas, int(FASE1_SEG / DUR_FASE1))
    t_fase1   = n_fase1 * DUR_FASE1
    n_fase2   = n_escenas - n_fase1
    dur_fase2 = (duracion_total_seg - t_fase1) / n_fase2 if n_fase2 > 0 else 0.0
    log.info(f"  Distribución: {n_fase1}×{DUR_FASE1:.0f}s + {n_fase2}×{dur_fase2:.1f}s"
             f" = {(t_fase1 + n_fase2*dur_fase2)/60:.1f} min")
    return [DUR_FASE1 if i < n_fase1 else dur_fase2 for i in range(n_escenas)]


# ══════════════════════════════════════════════════════════════════════════════
# SUBTÍTULOS ASS KARAOKE
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_ass(seg):
    seg = max(0.0, seg)
    h   = int(seg // 3600)
    m   = int((seg % 3600) // 60)
    s   = int(seg % 60)
    cs  = min(int(round((seg - int(seg)) * 100)), 99)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _build_karaoke_line(grupo_words, idx_activa):
    """Construye la línea ASS con palabra activa en AMARILLO, resto en BLANCO."""
    partes = []
    for k, w in enumerate(grupo_words):
        if k == idx_activa:
            partes.append(f"{{\\c{YELLOW}&}}{w}{{\\c{WHITE}&}}")
        else:
            partes.append(w)
    return " ".join(partes)


def generar_ass_whisper(words_timing, duracion_total, ass_path):
    """
    Karaoke con tiempos reales de Whisper word_timestamps.
    words_timing: [{"word": str, "start": float, "end": float}, ...]
    Grupos de 4 palabras en MAYÚSCULAS; la activa en amarillo.
    """
    os.makedirs(os.path.dirname(ass_path), exist_ok=True)

    words = [
        {"word": w["word"].strip().upper(), "start": w["start"], "end": w["end"]}
        for w in words_timing
        if w["word"].strip()
    ]

    events = []
    group_size = 4

    for i in range(0, len(words), group_size):
        grupo = words[i:i + group_size]
        grupo_txt = [w["word"] for w in grupo]
        for j, wi in enumerate(grupo):
            t_ini = wi["start"]
            t_fin = min(wi["end"], duracion_total)
            line  = _build_karaoke_line(grupo_txt, j)
            events.append(
                f"Dialogue: 0,{_fmt_ass(t_ini)},{_fmt_ass(t_fin)},"
                f"Default,,0,0,0,,{line}"
            )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER + "\n".join(events))

    log.info(f"  ✓ ASS Whisper: {len(events)} eventos")
    return ass_path


def generar_ass_guion(escenas, duracion_total, ass_path):
    """Fallback: tiempos proporcionales desde el texto del guión."""
    os.makedirs(os.path.dirname(ass_path), exist_ok=True)

    grupos = []
    for escena in escenas:
        for parrafo in [p.strip() for p in escena["texto_narrador"].split("\n") if p.strip()]:
            palabras = parrafo.upper().split()
            for i in range(0, len(palabras), 4):
                grupos.append(palabras[i:i + 4])

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
            line  = _build_karaoke_line(grupo, j)
            events.append(
                f"Dialogue: 0,{_fmt_ass(t_ini)},{_fmt_ass(t_fin)},"
                f"Default,,0,0,0,,{line}"
            )
        t = min(t + dur_g, duracion_total)

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER + "\n".join(events))

    log.info(f"  ✓ ASS guión: {len(events)} eventos")
    return ass_path


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE UPLOAD — Service Account
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_drive(file_path, folder_id, sa_json_raw):
    """Sube el MP4 a Google Drive usando Service Account."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2 import service_account

    # Acepta JSON crudo (str) o base64
    if sa_json_raw.strip().startswith("{"):
        sa_info = json.loads(sa_json_raw)
    else:
        sa_info = json.loads(base64.b64decode(sa_json_raw).decode())

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    filename      = os.path.basename(file_path)
    file_metadata = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(
        file_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024  # 10 MB por chunk
    )

    log.info(f"  Subiendo {filename} a Drive ({folder_id})...")
    req = service.files().create(
        body=file_metadata, media_body=media, fields="id,webViewLink"
    )
    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            log.info(f"  Drive: {int(status.progress() * 100)}%")

    fid = response.get("id")
    url = response.get("webViewLink", f"https://drive.google.com/file/d/{fid}/view")
    log.info(f"  ✓ Drive: {url}")
    return fid, url


# ══════════════════════════════════════════════════════════════════════════════
# MÚSICA AMBIENTE
# ══════════════════════════════════════════════════════════════════════════════

def conseguir_musica(workdir, musica_url=""):
    dest = os.path.join(workdir, "musica_ambiente.mp3")
    candidatos = ([musica_url] if musica_url else []) + MUSIC_CANDIDATES
    for url in candidatos:
        try:
            download_file(url, dest, "música ambiente")
            if os.path.getsize(dest) > 100_000:
                return dest
        except Exception as e:
            log.warning(f"  Música fallida ({url[:60]}...): {e}")
    log.warning("  Sin música disponible.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HANDLER RunPod
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── Validación básica ─────────────────────────────────────────────────────
    if not audio_url:
        return {"error": "Falta 'audio_url'", "success": False}
    if not imagenes_urls:
        return {"error": "Falta 'imagenes_urls'", "success": False}
    if not duracion_seg:
        return {"error": "Falta 'duracion_seg'", "success": False}

    # ── Workspace ─────────────────────────────────────────────────────────────
    workdir   = f"/tmp/{slug}"
    clips_dir = f"{workdir}/clips"
    subs_dir  = f"{workdir}/subtitulos"
    imgs_dir  = f"{workdir}/imagenes"

    for d in [workdir, clips_dir, subs_dir, imgs_dir]:
        os.makedirs(d, exist_ok=True)

    words_count = 0

    try:
        log.info("=" * 60)
        log.info(f"MONTAJE INICIO: {slug}")
        log.info(f"  Escenas    : {len(imagenes_urls)}")
        log.info(f"  Duración   : {duracion_seg/60:.1f} min")
        log.info("=" * 60)

        # ── PASO 1: Audio ─────────────────────────────────────────────────────
        log.info("\n[1/9] Descargando audio...")
        audio_path = f"{workdir}/audio.mp3"
        download_file(audio_url, audio_path, "audio MP3")

        # ── PASO 2: Imágenes ──────────────────────────────────────────────────
        log.info(f"\n[2/9] Descargando {len(imagenes_urls)} imágenes...")
        imagen_paths = []
        for i, url in enumerate(imagenes_urls, 1):
            dest = f"{imgs_dir}/escena_{i:03d}.png"
            try:
                download_file(url, dest, f"escena {i:03d}")
                imagen_paths.append(dest)
            except Exception as e:
                log.warning(f"  ⚠ escena {i:03d} fallida: {e}")
                imagen_paths.append(None)

        # ── PASO 3: Ken Burns ─────────────────────────────────────────────────
        log.info(f"\n[3/9] Ken Burns — {len(imagen_paths)} escenas...")
        duraciones = calcular_duraciones(len(imagen_paths), duracion_seg)
        clips      = []

        for i, (img, dur) in enumerate(zip(imagen_paths, duraciones), 1):
            out = f"{clips_dir}/kb_{i:03d}.mp4"
            if os.path.exists(out):
                clips.append(out)
                continue
            if not img or not os.path.exists(img):
                # Placeholder negro
                run_ffmpeg([
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"color=c=black:s={IMAGE_WIDTH}x{IMAGE_HEIGHT}:r={VIDEO_FPS}",
                    "-t", str(dur),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    out
                ], f"placeholder {i:03d}")
            else:
                apply_ken_burns(img, dur, out, i)
            clips.append(out)

        # ── PASO 4: Concatenar clips ──────────────────────────────────────────
        log.info(f"\n[4/9] Concatenando {len(clips)} clips...")
        lista_txt = f"{workdir}/lista_clips.txt"
        with open(lista_txt, "w") as f:
            for c in clips:
                if c and os.path.exists(c):
                    f.write(f"file '{c}'\n")

        video_sin_audio = f"{workdir}/video_sin_audio.mp4"
        run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", lista_txt,
            "-c:v", "libx264", "-preset", "fast",
            "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p",
            video_sin_audio
        ], "concat clips")

        # ── PASO 5: Whisper large → tiempos por palabra ───────────────────────
        log.info("\n[5/9] Transcribiendo con Whisper large...")
        words_timing = []
        try:
            result = whisper_model.transcribe(
                audio_path,
                language="es",
                word_timestamps=True,
                fp16=(_DEVICE == "cuda")
            )
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    raw = w.get("word", "").strip()
                    if raw:
                        words_timing.append({
                            "word":  raw,
                            "start": float(w["start"]),
                            "end":   float(w["end"]),
                        })
            words_count = len(words_timing)
            log.info(f"  ✓ Whisper: {words_count} palabras detectadas")
        except Exception as e:
            log.warning(f"  ⚠ Whisper falló: {e} — usando guión para subtítulos")

        # ── PASO 6: ASS karaoke ───────────────────────────────────────────────
        log.info("\n[6/9] Generando subtítulos karaoke ASS (MAYÚSCULAS + amarillo)...")
        ass_path = f"{subs_dir}/subtitulos.ass"
        if words_timing:
            generar_ass_whisper(words_timing, duracion_seg, ass_path)
        elif escenas:
            generar_ass_guion(escenas, duracion_seg, ass_path)
        else:
            log.warning("  Sin palabras ni guión — vídeo sin subtítulos")
            ass_path = None

        # ── PASO 7: Música ambiente ───────────────────────────────────────────
        log.info("\n[7/9] Música ambiente...")
        musica_path = conseguir_musica(workdir, musica_url)

        # ── PASO 8: Montaje final (video + audio + subtítulos + música) ───────
        log.info("\n[8/9] Montaje final FFmpeg...")
        video_final = f"{workdir}/{slug}_FINAL.mp4"

        # libass en Linux usa path tal cual (no escapado como en Windows)
        ass_for_ffmpeg = ass_path.replace("\\", "/") if ass_path else None

        if ass_for_ffmpeg and musica_path:
            filter_complex = (
                "[1:a]volume=1.0[narr];"
                "[2:a]volume=0.12[amb];"
                "[narr][amb]amix=inputs=2:duration=first[aout];"
                f"[0:v]ass='{ass_for_ffmpeg}'[vout]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", video_sin_audio,
                "-i", audio_path,
                "-stream_loop", "-1", "-i", musica_path,
                "-filter_complex", filter_complex,
                "-map", "[vout]", "-map", "[aout]",
                "-c:a", "aac", "-b:a", "192k",
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p",
                "-t", str(duracion_seg),
                video_final
            ]
        elif ass_for_ffmpeg:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_sin_audio,
                "-i", audio_path,
                "-vf", f"ass='{ass_for_ffmpeg}'",
                "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k",
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p",
                "-t", str(duracion_seg),
                video_final
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_sin_audio,
                "-i", audio_path,
                "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k",
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p",
                "-t", str(duracion_seg),
                video_final
            ]

        run_ffmpeg(cmd, "montaje final")

        # Verificar duración real del MP4
        probe = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_final
        ], capture_output=True, text=True)
        dur_final = float(probe.stdout.strip())
        size_mb   = os.path.getsize(video_final) / 1_048_576

        log.info(f"\n  ✓ VÍDEO FINAL: {dur_final/60:.1f} min — {size_mb:.0f} MB")

        # ── PASO 9: Subida a Google Drive ─────────────────────────────────────
        drive_file_id = None
        drive_url     = None

        if folder_id and sa_json:
            log.info("\n[9/9] Subiendo a Google Drive...")
            drive_file_id, drive_url = upload_to_drive(video_final, folder_id, sa_json)
        else:
            log.info("\n[9/9] Sin credenciales Drive — vídeo en /tmp (recuperar via SSH)")

        elapsed = (time.time() - t0) / 60
        log.info(f"\nTiempo total: {elapsed:.1f} min")

        return {
            "success":            True,
            "drive_file_id":      drive_file_id,
            "drive_url":          drive_url,
            "duracion_final_seg": round(dur_final, 1),
            "size_mb":            round(size_mb, 1),
            "elapsed_min":        round(elapsed, 1),
            "whisper_words":      words_count,
        }

    except Exception as exc:
        log.exception(f"ERROR: {exc}")
        return {"success": False, "error": str(exc)}

    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
