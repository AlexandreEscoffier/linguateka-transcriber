from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from openai import OpenAI
import subprocess
import uuid
import os

app = FastAPI()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Dossier de téléchargement
DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)


# -------------------------
# MODELES DE REQUETE / REPONSE
# -------------------------

class TranscribeRequest(BaseModel):
    url: str

class Segment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float

class TranscribeResponse(BaseModel):
    transcript: str
    segments: list[Segment]


# -------------------------
# FONCTION : télécharger la vidéo
# -------------------------

def download_instagram_video(url: str) -> Path:
    """Télécharge une vidéo Instagram avec yt-dlp et renvoie le chemin local."""
    uid = uuid.uuid4().hex
    template = DOWNLOAD_DIR / f"{uid}.%(ext)s"

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", str(template),
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("yt-dlp error: " + result.stderr)

    # Trouver le fichier téléchargé
    videos = list(DOWNLOAD_DIR.glob(f"{uid}.*"))
    if not videos:
        raise FileNotFoundError("Aucun fichier téléchargé")

    return videos[0]


# -------------------------
# FONCTION : Transcrire avec diarisation
# -------------------------

def transcribe_with_diarization(video_path: Path) -> tuple[str, list[dict]]:
    with open(video_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=f,
            response_format="diarized_json",
            chunking_strategy="auto",
        )

    lines = []
    segments = []

    for seg in tr.segments:
        speaker = seg.speaker
        text = seg.text.strip()
        if not text:
            continue

        lines.append(f"{speaker}: {text}")

        segments.append({
            "speaker": speaker,
            "text": text,
            "start": seg.start,
            "end": seg.end,
        })

    return "\n".join(lines), segments


# -------------------------
# ROUTE PRINCIPALE
# -------------------------

@app.post("/transcribe-instagram", response_model=TranscribeResponse)
async def transcribe_instagram(body: TranscribeRequest):

    try:
        video_path = download_instagram_video(body.url)
        transcript_text, segs = transcribe_with_diarization(video_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Nettoyage du fichier téléchargé
        try:
            if "video_path" in locals() and video_path.exists():
                video_path.unlink()
        except:
            pass

    return TranscribeResponse(
        transcript=transcript_text,
        segments=[Segment(**s) for s in segs]
    )