import time
import subprocess
import uuid
import os
import json
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from openai import OpenAI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI()

# -------------------------
# PROMETHEUS METRICS
# -------------------------

REQUEST_COUNT = Counter(
    "fastapi_requests_total",
    "Total HTTP requests",
    ["endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "fastapi_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"]
)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


# Railway : TRANSCRIPTION_MODEL=gpt-transcribe (rapide, sans diarisation)
# ou TRANSCRIPTION_MODEL=gpt-4o-transcribe-diarize (avec diarisation)
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "gpt-transcribe").strip()
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "openai").strip().lower()
SUPPORTED_TRANSCRIPTION_MODELS = {
    "gpt-transcribe",
    "gpt-4o-transcribe-diarize",
}

if TRANSCRIPTION_MODEL not in SUPPORTED_TRANSCRIPTION_MODELS:
    raise RuntimeError(
        f"TRANSCRIPTION_MODEL non pris en charge : {TRANSCRIPTION_MODEL!r}. "
        f"Choisir l'un des modeles : {sorted(SUPPORTED_TRANSCRIPTION_MODELS)}"
    )


def transcribe_video(video_path: Path) -> tuple[str, list[dict]]:
    """Renvoie (transcript, segments), quel que soit le modele.

    gpt-transcribe fournit le texte seul : segments=[] (aucun locuteur invente).
    Le modele diarize renvoie le texte avec les locuteurs et leurs segments.
    """
    
    if TRANSCRIPTION_PROVIDER == "deepgram":
        result = transcribe_with_deepgram(video_path)
        return result["transcript"], result["segments"]
    
    if TRANSCRIPTION_MODEL == "gpt-transcribe":
        options = {
            "model": "gpt-transcribe",
            "response_format": "json",
        }
        # Optionnel : TRANSCRIPTION_LANGUAGE=pl sur Railway pour tester le polonais.
        # Par defaut, laisser vide pour autodetection et plusieurs langues.
        lang = os.getenv("TRANSCRIPTION_LANGUAGE", "").strip().lower()
        if lang:
            options["languages"] = [lang]

        with open(video_path, "rb") as f:
            tr = client.audio.transcriptions.create(file=f, **options)

        transcript_text = (tr.text or "").strip()
        if not transcript_text:
            raise RuntimeError("Le modele gpt-transcribe n'a retourne aucun texte.")
        return transcript_text, []

    # Modele historique, avec identification des locuteurs.
    return transcribe_with_diarization(video_path)


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
        segments.append({
            "speaker": speaker,
            "text": text,
            "start": seg.start,
            "end": seg.end,
        })

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
    start_time = time.time()
    status_code = "200"
    video_path: Path | None = None

    try:
        if not body.url.strip():
            status_code = "400"
            raise HTTPException(status_code=400, detail="URL is empty.")

        
        # 1. Mesurer le téléchargement
        download_start = time.perf_counter()

        video_path = download_instagram_video(body.url)

        download_duration = time.perf_counter() - download_start

        print(
            f"[LINGUATEKA] Téléchargement yt-dlp terminé : "
            f"{download_duration:.2f} secondes",
            flush=True
        )

        # 2. Transcrire avec le modele choisi dans Railway
        print(f"[LINGUATEKA] Modele utilise : {TRANSCRIPTION_MODEL}", flush=True)

        transcription_start = time.perf_counter()
        transcript_text, segs = transcribe_video(video_path)
        transcription_duration = time.perf_counter() - transcription_start

        print(
            f"[LINGUATEKA] Transcription {TRANSCRIPTION_MODEL} : "
            f"{transcription_duration:.2f} secondes",
            flush=True,
        )
        print(
            f"[LINGUATEKA] TOTAL (telechargement + transcription) : "
            f"{download_duration + transcription_duration:.2f} secondes",
            flush=True,
        )
        if not segs:
            print(
                "[LINGUATEKA] Pas de diarisation : segments=[] ; "
                "transcript contient le texte integral.",
                flush=True,
            )

        return TranscribeResponse(
            transcript=transcript_text,
            segments=[Segment(**s) for s in segs],
        )

    except HTTPException as e:
        status_code = str(e.status_code)
        raise e

    except Exception as e:
        status_code = "400"
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        REQUEST_COUNT.labels(
            endpoint="/transcribe-instagram",
            http_status=status_code
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/transcribe-instagram"
        ).observe(time.time() - start_time)

        try:
            if video_path and video_path.exists():
                video_path.unlink()
        except Exception:
            pass


@app.post("/translate", response_model=TranslateResponse)
async def translate(body: TranslateRequest):
    start_time = time.time()
    status_code = "200"

    try:
        if not body.text.strip():
            status_code = "400"
            raise HTTPException(status_code=400, detail="Text is empty.")

        requested = [(t or "").lower().strip() for t in (body.targets or [])]
        allowed = [c for c in requested if c in LANG_NAMES]

        if not allowed:
            status_code = "400"
            raise HTTPException(
                status_code=400,
                detail="No valid target languages provided."
            )

        translations = translate_text_with_openai(body.text, allowed)
        return TranslateResponse(translations=translations)

    except HTTPException as e:
        status_code = str(e.status_code)
        raise e

    except Exception as e:
        status_code = "500"
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        REQUEST_COUNT.labels(
            endpoint="/translate",
            http_status=status_code
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/translate"
        ).observe(time.time() - start_time)


@app.post("/thumbnail", response_model=ThumbnailResponse)
async def thumbnail(body: ThumbnailRequest):
    start_time = time.time()
    status_code = "200"

    try:
        thumb_url = fetch_thumbnail_url(body.url)
        return ThumbnailResponse(thumbnail_url=thumb_url)

    except Exception as e:
        status_code = "400"
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        REQUEST_COUNT.labels(
            endpoint="/thumbnail",
            http_status=status_code
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/thumbnail"
        ).observe(time.time() - start_time)
        


def transcribe_with_deepgram(video_path):
    api_key = os.environ["DEEPGRAM_API_KEY"]

    params = {
        "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
        "language": "pl",
        "diarize_model": "latest",
        "utterances": "true",
        "punctuate": "true",
        "smart_format": "true",
    }

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "video/mp4",
    }

    start = time.perf_counter()

    with open(video_path, "rb") as video:
        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            data=video,
            timeout=(15, 180),
        )

    response.raise_for_status()
    data = response.json()
    
    words = (
    data["results"]["channels"][0]
    ["alternatives"][0].get("words", [])
)

    for word in words:
        print(
            f"[DIARIZATION] "
            f"{word.get('start')}s "
            f"speaker={word.get('speaker')} "
            f"word={word.get('word')}",
            flush=True
        )

    print(
        f"Deepgram : {time.perf_counter() - start:.2f}s"
    )

    results = data.get("results", {})
    alternatives = (
        results.get("channels", [{}])[0]
        .get("alternatives", [{}])
    )

    raw_transcript = alternatives[0].get(
        "transcript", ""
    )

    segments = []
    lines = []

    for utterance in results.get("utterances", []):
        speaker_id = utterance.get("speaker")

        speaker = (
            chr(65 + speaker_id)
            if isinstance(speaker_id, int)
            and 0 <= speaker_id < 26
            else "?"
        )

        text = utterance.get("transcript", "").strip()

        segment = {
            "speaker": speaker,
            "start": utterance["start"],
            "end": utterance["end"],
            "text": text,
            "words": [
                {
                    "word": word.get(
                        "punctuated_word",
                        word.get("word", "")
                    ),
                    "start": word["start"],
                    "end": word["end"],
                }
                for word in utterance.get("words", [])
            ],
        }

        segments.append(segment)
        lines.append(f"{speaker}: {text}")

    return {
        "transcript": "\n".join(lines)
            if lines else raw_transcript,
        "raw_transcript": raw_transcript,
        "segments": segments,
    }
