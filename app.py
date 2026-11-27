import time
import subprocess
import uuid
import os
import json
import requests
from yt_dlp_journal import run_logged

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    detect_language: bool = False


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = Field(default_factory=list)


class TranscribeResponse(BaseModel):
    language_code: str | None = None
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

    # Scribe only needs audio. Some Reels serve video and sound separately.
    media_format = "bestaudio/best/best*" if TRANSCRIPTION_PROVIDER == "elevenlabs" else "mp4"
    cmd = ["yt-dlp", "--no-playlist", "-f", media_format, "-o", str(template), "--", url]
    result = run_logged(cmd, "audio", url)

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
    """Renvoie (transcript, segments) selon le fournisseur."""
    
    # ElevenLabs Scribe v3
    if TRANSCRIPTION_PROVIDER == "elevenlabs":
        result = transcribe_with_elevenlabs(video_path)
        return result["transcript"], result["segments"]

    # 1. AssemblyAI
    if TRANSCRIPTION_PROVIDER == "assemblyai":
        result = transcribe_with_assemblyai(video_path)
        return result["transcript"], result["segments"]

    # 2. Deepgram
    if TRANSCRIPTION_PROVIDER == "deepgram":
        result = transcribe_with_deepgram(video_path)
        return result["transcript"], result["segments"]

    # 3. OpenAI
    if TRANSCRIPTION_PROVIDER == "openai":

        if TRANSCRIPTION_MODEL == "gpt-transcribe":
            options = {
                "model": "gpt-transcribe",
                "response_format": "json",
            }

            lang = os.getenv(
                "TRANSCRIPTION_LANGUAGE", ""
            ).strip().lower()

            if lang:
                options["languages"] = [lang]

            with open(video_path, "rb") as f:
                tr = client.audio.transcriptions.create(
                    file=f,
                    **options
                )

            transcript_text = (tr.text or "").strip()

            if not transcript_text:
                raise RuntimeError(
                    "gpt-transcribe n'a retourne aucun texte."
                )

            return transcript_text, []

        # Ancien modèle OpenAI avec diarisation
        return transcribe_with_diarization(video_path)

    raise ValueError(
        f"Fournisseur inconnu : {TRANSCRIPTION_PROVIDER}"
    )



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
    result = run_logged(cmd, "thumbnail", url)

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
        language_code = None
        if body.detect_language and TRANSCRIPTION_PROVIDER == "elevenlabs":
            result = transcribe_with_elevenlabs(video_path, detect_language=True)
            transcript_text, segs = result["transcript"], result["segments"]
            language_code = result.get("language_code")
        else:
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
            language_code=language_code,
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
    
    
def transcribe_with_assemblyai(video_path: Path):
    api_key = os.environ["ASSEMBLYAI_API_KEY"]
    base_url = "https://api.assemblyai.com/v2"
    headers = {"authorization": api_key}

    start_time = time.perf_counter()

    # 1. Envoyer la vidéo à AssemblyAI
    with open(video_path, "rb") as video:
        response = requests.post(
            f"{base_url}/upload",
            headers=headers,
            data=video,
            timeout=(15, 120),
        )

    response.raise_for_status()
    upload_url = response.json()["upload_url"]

    # 2. Demander la transcription polonaise avec diarisation
    payload = {
        "audio_url": upload_url,
        "speech_models": ["universal-2"],
        "language_code": "pl",
        "speaker_labels": True,
    }

    response = requests.post(
        f"{base_url}/transcript",
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    transcript_id = response.json()["id"]

    # 3. Attendre que le traitement soit terminé
    deadline = time.monotonic() + 180

    while time.monotonic() < deadline:
        response = requests.get(
            f"{base_url}/transcript/{transcript_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        if result["status"] == "completed":
            break

        if result["status"] == "error":
            raise RuntimeError(
                f"AssemblyAI: {result.get('error')}"
            )

        time.sleep(2)
    else:
        raise TimeoutError(
            f"AssemblyAI: délai dépassé, job {transcript_id}"
        )

    # 4. Convertir les données pour Linguateka
    segments = []
    lines = []

    for utterance in result.get("utterances") or []:
        speaker = utterance.get("speaker") or "?"
        text = (utterance.get("text") or "").strip()

        if not text:
            continue

        words = [
            {
                "word": word["text"],
                "start": word["start"] / 1000,
                "end": word["end"] / 1000,
            }
            for word in utterance.get("words") or []
        ]

        segments.append({
            "speaker": speaker,
            "text": text,
            "start": utterance["start"] / 1000,
            "end": utterance["end"] / 1000,
            "words": words,
        })

        lines.append(f"{speaker}: {text}")

    transcript = "\n".join(lines)

    if not transcript:
        raise RuntimeError(
            "AssemblyAI n'a retourné aucun passage de parole."
        )

    duration = time.perf_counter() - start_time
    print(
        f"[ASSEMBLYAI] Terminé en {duration:.2f}s "
        f"- {len(segments)} segments",
        flush=True,
    )

    return {
        "transcript": transcript,
        "segments": segments,
    }



def transcribe_with_elevenlabs(video_path: Path, *, detect_language: bool = False):
    from tempfile import TemporaryDirectory
    from imageio_ffmpeg import get_ffmpeg_exe
    import wave

    api_key = os.environ["ELEVENLABS_API_KEY"]
    model = os.getenv("ELEVENLABS_MODEL", "scribe_v3")
    language = os.getenv("TRANSCRIPTION_LANGUAGE", "pl")

    start = time.perf_counter()

    headers = {
        "xi-api-key": api_key
    }

    payload = {
        "model_id": model,
        "diarize": "true",
        "timestamps_granularity": "word",
        "tag_audio_events": "false",
        "no_verbatim": "false",
    }

    if not detect_language and language:
        payload["language_code"] = language

    with TemporaryDirectory(prefix="linguateka-audio-") as directory:
        audio_path = Path(directory) / "audio.wav"
        conversion = subprocess.run(
            [get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(video_path),
             "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(audio_path)],
            capture_output=True, timeout=90,
        )
        if conversion.returncode:
            raise RuntimeError("source_audio_unreadable")
        with wave.open(str(audio_path), "rb") as wav:
            if not wav.getnframes():
                raise RuntimeError("source_audio_unreadable")
        with audio_path.open("rb") as audio:
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers=headers, data=payload,
                files={"file": ("audio.wav", audio, "audio/wav")},
                timeout=(20, 240),
            )

    if not response.ok:
        raise RuntimeError(
            f"ElevenLabs HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    result = response.json()

    # Les mots possèdent des horodatages en secondes
    # et des identifiants de locuteur : speaker_0, speaker_1...
    segments = []
    lines = []
    speaker_labels = {}
    current_speaker = None
    pieces = []
    current_words = []

    def label_for(speaker_id):
        if speaker_id == "unknown":
            return "?"
        if speaker_id not in speaker_labels:
            index = len(speaker_labels)
            speaker_labels[speaker_id] = (
                chr(65 + index)
                if index < 26 else f"S{index + 1}"
            )
        return speaker_labels[speaker_id]

    def flush_segment():
        if not current_words:
            return

        text = "".join(pieces).strip()
        if not text:
            text = " ".join(
                w["word"] for w in current_words
            )

        speaker = label_for(current_speaker)
        segments.append({
            "speaker": speaker,
            "text": text,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "words": current_words.copy(),
        })
        lines.append(f"{speaker}: {text}")

    for token in result.get("words", []):
        kind = token.get("type")

        if kind == "spacing":
            if pieces:
                pieces.append(token.get("text", " "))
            continue

        if kind != "word":
            continue

        speaker_id = (
            token.get("speaker_id")
            or current_speaker
            or "unknown"
        )

        if (
            current_speaker is not None
            and speaker_id != current_speaker
        ):
            flush_segment()
            pieces = []
            current_words = []

        current_speaker = speaker_id
        word_text = token.get("text", "")

        if (
            pieces
            and not pieces[-1].endswith((" ", "\n"))
            and not word_text.startswith(
                (",", ".", "!", "?", ";", ":")
            )
        ):
            pieces.append(" ")

        pieces.append(word_text)

        current_words.append({
            "word": word_text,
            "start": float(token["start"]),
            "end": float(token["end"]),
        })

    flush_segment()

    transcript = (
        "\n".join(lines)
        if lines else (result.get("text") or "").strip()
    )

    if not transcript:
        raise RuntimeError(
            "ElevenLabs n'a retourné aucun texte."
        )

    duration = time.perf_counter() - start

    print(
        f"[ELEVENLABS] {model} | "
        f"{duration:.2f}s | "
        f"{len(segments)} segments",
        flush=True,
    )

    return {
        "language_code": result.get("language_code"),
        "transcript": transcript,
        "segments": segments,
    }


from lesson_vocabulary import router as vocabulary_router
app.include_router(vocabulary_router)
