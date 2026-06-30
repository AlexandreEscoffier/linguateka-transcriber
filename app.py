from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from openai import OpenAI
import subprocess
import uuid
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# -------------------------
# MODELES
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

class TranslateRequest(BaseModel):
    text: str
    targets: list[str]

class TranslateResponse(BaseModel):
    translations: dict[str, str]

class ThumbnailRequest(BaseModel):
    url: str

class ThumbnailResponse(BaseModel):
    thumbnail_url: str

# -------------------------
# HELPERS
# -------------------------

LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "pl": "Polish",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "no": "Norwegian",
    "sv": "Swedish",
    "fi": "Finnish",
    "nl": "Dutch",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sk": "Slovak",
    "ro": "Romanian",
    "el": "Greek",
    "sr": "Serbian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}

def download_instagram_video(url: str) -> Path:
    uid = uuid.uuid4().hex
    template = DOWNLOAD_DIR / f"{uid}.%(ext)s"

    cmd = ["yt-dlp", "-f", "mp4", "-o", str(template), url]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError("yt-dlp error: " + result.stderr)

    videos = list(DOWNLOAD_DIR.glob(f"{uid}.*"))
    if not videos:
        raise FileNotFoundError("Aucun fichier téléchargé")

    return videos[0]

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
        text = (seg.text or "").strip()
        if not text:
            continue

        lines.append(f"{speaker}: {text}")
        segments.append(
            {"speaker": speaker, "text": text, "start": seg.start, "end": seg.end}
        )

    return "\n".join(lines), segments

def translate_text_with_openai(text: str, targets: list[str]) -> dict[str, str]:
    translations: dict[str, str] = {}

    for code in targets:
        code_lower = (code or "").lower().strip()
        lang_label = LANG_NAMES.get(code_lower)
        if not lang_label:
            continue

        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. "
                        "You translate text precisely while keeping the meaning natural."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Translate the following text into {lang_label}. "
                        "Only answer with the translated text, nothing else.\n\n"
                        f"{text}"
                    ),
                },
            ],
        )

        translations[code_lower] = completion.choices[0].message.content.strip()

    return translations

def fetch_thumbnail_url(url: str) -> str:
    cmd = ["yt-dlp", "-J", url]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError("yt-dlp error: " + result.stderr)

    info = json.loads(result.stdout)

    thumb = info.get("thumbnail")
    if not thumb:
        thumbs = info.get("thumbnails") or []
        if thumbs:
            thumb = thumbs[-1].get("url")

    if not thumb:
        raise RuntimeError("No thumbnail found for this URL.")

    return thumb

# -------------------------
# ROUTES
# -------------------------

@app.post("/transcribe-instagram", response_model=TranscribeResponse)
async def transcribe_instagram(body: TranscribeRequest):
    if not body.url.strip():
        raise HTTPException(status_code=400, detail="URL is empty.")

    video_path: Path | None = None
    try:
        video_path = download_instagram_video(body.url)
        transcript_text, segs = transcribe_with_diarization(video_path)
        return TranscribeResponse(
            transcript=transcript_text,
            segments=[Segment(**s) for s in segs],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            if video_path and video_path.exists():
                video_path.unlink()
        except:
            pass

@app.post("/translate", response_model=TranslateResponse)
async def translate(body: TranslateRequest):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    requested = [(t or "").lower().strip() for t in (body.targets or [])]
    allowed = [c for c in requested if c in LANG_NAMES]

    if not allowed:
        raise HTTPException(status_code=400, detail="No valid target languages provided.")

    try:
        translations = translate_text_with_openai(body.text, allowed)
        return TranslateResponse(translations=translations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/thumbnail", response_model=ThumbnailResponse)
async def thumbnail(body: ThumbnailRequest):
    try:
        thumb_url = fetch_thumbnail_url(body.url)
        return ThumbnailResponse(thumbnail_url=thumb_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
