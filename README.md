# biblia-montaje — RunPod Serverless Worker

Worker de montaje de vídeo bíblico para RunPod Serverless GPU.

## Stack

- **FFmpeg** — Ken Burns, concat, subtítulos, mezcla audio
- **Whisper large** (bakeado en imagen) — tiempos word-by-word para karaoke
- **Subtítulos ASS** — karaoke amarillo MAYÚSCULAS, grupos de 4 palabras
- **Google Drive** — subida vía Service Account

## Input JSON

```json
{
  "slug":          "mi_video",
  "audio_url":     "https://...",
  "imagenes_urls": ["https://...png", "..."],
  "escenas":       [{"numero": 1, "texto_narrador": "..."}],
  "duracion_seg":  1200.5,
  "musica_url":    "https://... (opcional)",
  "drive_folder_id": "1ABC...",
  "gdrive_sa_json":  "{ ... Service Account JSON ... }"
}
```

## Output JSON

```json
{
  "success":            true,
  "drive_file_id":      "1XYZ...",
  "drive_url":          "https://drive.google.com/file/d/.../view",
  "duracion_final_seg": 1200.1,
  "size_mb":            850.3,
  "elapsed_min":        22.4,
  "whisper_words":      3120
}
```

## Despliegue

1. El workflow de GitHub Actions sube la imagen a `kuekuatsu17/biblia-montaje:latest`
2. En RunPod: **New Endpoint → Docker image** → `kuekuatsu17/biblia-montaje:latest`
3. GPU recomendada: RTX 3090 o A40 (Whisper large requiere ~3 GB VRAM)

## Secrets requeridos en GitHub

| Secret              | Valor                          |
|---------------------|--------------------------------|
| `DOCKERHUB_USERNAME`| tu usuario Docker Hub          |
| `DOCKERHUB_TOKEN`   | Access Token de Docker Hub     |

## Service Account Google Drive

`biblia-montaje-worker@buscador-de-videos-496106.iam.gserviceaccount.com`

Permisos necesarios: **Editor** en la carpeta Drive de destino.
