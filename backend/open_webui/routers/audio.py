import hashlib
import json
import logging
import os
import uuid
import html
import base64
import mimetypes
import struct
from pydub import AudioSegment
from pydub.silence import split_on_silence
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fnmatch import fnmatch
import aiohttp
import aiofiles
import requests

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


from open_webui.utils.misc import strict_match_mime_type
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.headers import include_user_info_headers
from open_webui.config import CACHE_DIR

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    ENV,
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    ENABLE_FORWARD_USER_INFO_HEADERS,
)


router = APIRouter()

# Constants
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
AZURE_MAX_FILE_SIZE_MB = 200
AZURE_MAX_FILE_SIZE = AZURE_MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

log = logging.getLogger(__name__)

VALID_STT_ENGINES = {"openai", "azure", "web"}

GEMINI_TTS_MODELS = [
    {"id": "gemini-3.1-flash-tts-preview"},
    {"id": "gemini-2.5-flash-preview-tts"},
    {"id": "gemini-2.5-pro-preview-tts"},
]

GEMINI_TTS_VOICES = {
    "Zephyr": "Zephyr - Bright",
    "Puck": "Puck - Upbeat",
    "Charon": "Charon - Informative",
    "Kore": "Kore - Firm",
    "Fenrir": "Fenrir - Excitable",
    "Leda": "Leda - Youthful",
    "Orus": "Orus - Firm",
    "Aoede": "Aoede - Breezy",
    "Callirrhoe": "Callirrhoe - Easy-going",
    "Autonoe": "Autonoe - Bright",
    "Enceladus": "Enceladus - Breathy",
    "Iapetus": "Iapetus - Clear",
    "Umbriel": "Umbriel - Easy-going",
    "Algieba": "Algieba - Smooth",
    "Despina": "Despina - Smooth",
    "Erinome": "Erinome - Clear",
    "Algenib": "Algenib - Gravelly",
    "Rasalgethi": "Rasalgethi - Informative",
    "Laomedeia": "Laomedeia - Upbeat",
    "Achernar": "Achernar - Soft",
    "Alnilam": "Alnilam - Firm",
    "Schedar": "Schedar - Even",
    "Gacrux": "Gacrux - Mature",
    "Pulcherrima": "Pulcherrima - Forward",
    "Achird": "Achird - Friendly",
    "Zubenelgenubi": "Zubenelgenubi - Casual",
    "Vindemiatrix": "Vindemiatrix - Gentle",
    "Sadachbia": "Sadachbia - Lively",
    "Sadaltager": "Sadaltager - Knowledgeable",
    "Sulafat": "Sulafat - Higher-pitched",
}

GEMINI_TTS_STYLE_PROMPTS = {
    "Vocal Smile": 'The "Vocal Smile": The soft palate is raised to keep the tone bright, sunny, and explicitly inviting.',
    "Newscaster": "Newscaster: Professional, authoritative, clear articulation with standard broadcast cadence.",
    "Whisper": "Whisper: Intimate, breathy, close-to-mic proximity effect.",
    "Empathetic": "Empathetic: Warm, understanding, soft tone with gentle inflections.",
    "Promo/Hype": "Promo/Hype: High energy, punchy consonants, elongated vowels on excitement words.",
    "Deadpan": "Deadpan: Flat affect, minimal pitch variation, dry delivery.",
}

GEMINI_TTS_PACE_PROMPTS = {
    "Natural": "Natural conversational pace.",
    "Rapid Fire": "Rapid Fire: Fast, energetic, no dead air. Sentences overlap slightly.",
    "The Drift": "The Drift: Slow, liquid, zero urgency. Long pauses for breath.",
    "Staccato": "Staccato: Short, clipped sentences with distinct pauses between words.",
}

SPEECH_CACHE_DIR = CACHE_DIR / "audio" / "speech"
SPEECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


##########################################
#
# Utility functions
#
##########################################

from pydub import AudioSegment
from pydub.utils import mediainfo


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample = 16
    rate = 24000

    for param in mime_type.split(";"):
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif param.lower().startswith("audio/l"):
            try:
                bits_per_sample = int(param.lower().split("audio/l", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def build_gemini_tts_prompt(
    scene: str,
    sample_context: str,
    transcript: str,
    style: str = "",
    pace: str = "",
    accent: str = "",
) -> str:
    prompt_parts = []
    director_notes = []

    if style.strip():
        style_prompt = GEMINI_TTS_STYLE_PROMPTS.get(style.strip(), style.strip())
        director_notes.append(f"Style: {style_prompt}")

    if pace.strip():
        pace_prompt = GEMINI_TTS_PACE_PROMPTS.get(pace.strip(), pace.strip())
        director_notes.append(f"Pace: {pace_prompt}")

    if accent.strip():
        director_notes.append(f"Accent: {accent.strip()}.")

    if director_notes:
        prompt_parts.append(
            "Read the following transcript based on the director's note.\n\n"
            "# Director's note\n"
            + " ".join(director_notes)
        )

    if scene.strip():
        prompt_parts.append(f"## Scene:\n{scene.strip()}")

    if sample_context.strip():
        prompt_parts.append(f"## Sample Context:\n{sample_context.strip()}")

    prompt_parts.append(f"## Transcript:\n{transcript}")

    return "\n\n".join(prompt_parts)


def extract_gemini_audio(response_data):
    responses = response_data if isinstance(response_data, list) else [response_data]

    for response in responses:
        for candidate in response.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data and inline_data.get("data"):
                    audio_data = base64.b64decode(inline_data["data"])
                    mime_type = inline_data.get("mimeType") or inline_data.get(
                        "mime_type", "audio/L16;rate=24000"
                    )
                    return audio_data, mime_type

    return None, None


def prepare_gemini_audio_file(audio_data: bytes, mime_type: str) -> tuple[bytes, str, str]:
    file_extension = mimetypes.guess_extension(mime_type.split(";", 1)[0])

    if file_extension is None:
        return convert_to_wav(audio_data, mime_type), ".wav", "audio/wav"

    return audio_data, file_extension, mime_type


def is_audio_conversion_required(file_path):
    """
    Check if the given audio file needs conversion to mp3.
    """
    SUPPORTED_FORMATS = {"flac", "m4a", "mp3", "mp4", "mpeg", "wav", "webm"}

    if not os.path.isfile(file_path):
        log.error(f"File not found: {file_path}")
        return False

    try:
        info = mediainfo(file_path)
        codec_name = info.get("codec_name", "").lower()
        codec_type = info.get("codec_type", "").lower()
        codec_tag_string = info.get("codec_tag_string", "").lower()

        if codec_name == "aac" and codec_type == "audio" and codec_tag_string == "mp4a":
            # File is AAC/mp4a audio, recommend mp3 conversion
            return True

        # If the codec name is in the supported formats
        if codec_name in SUPPORTED_FORMATS:
            return False

        return True
    except Exception as e:
        log.error(f"Error getting audio format: {e}")
        return False


def convert_audio_to_mp3(file_path):
    """Convert audio file to mp3 format."""
    try:
        output_path = os.path.splitext(file_path)[0] + ".mp3"
        audio = AudioSegment.from_file(file_path)
        audio.export(output_path, format="mp3")
        log.info(f"Converted {file_path} to {output_path}")
        return output_path
    except Exception as e:
        log.error(f"Error converting audio file: {e}")
        return None


def normalize_stt_engine(engine: str) -> str:
    return engine if engine in VALID_STT_ENGINES else "openai"


##########################################
#
# Audio API
#
##########################################


class TTSConfigForm(BaseModel):
    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: str
    OPENAI_PARAMS: Optional[dict] = None
    GEMINI_API_BASE_URL: str = "https://generativelanguage.googleapis.com"
    GEMINI_API_KEY: str = ""
    GEMINI_PARAMS: Optional[dict] = None
    GEMINI_SCENE: str = ""
    GEMINI_SAMPLE_CONTEXT: str = ""
    GEMINI_STYLE: str = ""
    GEMINI_PACE: str = ""
    GEMINI_ACCENT: str = ""
    GEMINI_TEMPERATURE: float = 1
    API_KEY: str
    ENGINE: str
    MODEL: str
    VOICE: str
    SPLIT_ON: str
    AZURE_SPEECH_REGION: str
    AZURE_SPEECH_BASE_URL: str
    AZURE_SPEECH_OUTPUT_FORMAT: str


class STTConfigForm(BaseModel):
    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: str
    ENGINE: str
    MODEL: str
    SUPPORTED_CONTENT_TYPES: list[str] = []
    AZURE_API_KEY: str
    AZURE_REGION: str
    AZURE_LOCALES: str
    AZURE_BASE_URL: str
    AZURE_MAX_SPEAKERS: str


class AudioConfigUpdateForm(BaseModel):
    tts: TTSConfigForm
    stt: STTConfigForm


@router.get("/config")
async def get_audio_config(request: Request, user=Depends(get_admin_user)):
    return {
        "tts": {
            "OPENAI_API_BASE_URL": request.app.state.config.TTS_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.TTS_OPENAI_API_KEY,
            "OPENAI_PARAMS": request.app.state.config.TTS_OPENAI_PARAMS,
            "GEMINI_API_BASE_URL": request.app.state.config.TTS_GEMINI_API_BASE_URL,
            "GEMINI_API_KEY": request.app.state.config.TTS_GEMINI_API_KEY,
            "GEMINI_PARAMS": request.app.state.config.TTS_GEMINI_PARAMS,
            "GEMINI_SCENE": request.app.state.config.TTS_GEMINI_SCENE,
            "GEMINI_SAMPLE_CONTEXT": request.app.state.config.TTS_GEMINI_SAMPLE_CONTEXT,
            "GEMINI_STYLE": request.app.state.config.TTS_GEMINI_STYLE,
            "GEMINI_PACE": request.app.state.config.TTS_GEMINI_PACE,
            "GEMINI_ACCENT": request.app.state.config.TTS_GEMINI_ACCENT,
            "GEMINI_TEMPERATURE": request.app.state.config.TTS_GEMINI_TEMPERATURE,
            "API_KEY": request.app.state.config.TTS_API_KEY,
            "ENGINE": request.app.state.config.TTS_ENGINE,
            "MODEL": request.app.state.config.TTS_MODEL,
            "VOICE": request.app.state.config.TTS_VOICE,
            "SPLIT_ON": request.app.state.config.TTS_SPLIT_ON,
            "AZURE_SPEECH_REGION": request.app.state.config.TTS_AZURE_SPEECH_REGION,
            "AZURE_SPEECH_BASE_URL": request.app.state.config.TTS_AZURE_SPEECH_BASE_URL,
            "AZURE_SPEECH_OUTPUT_FORMAT": request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT,
        },
        "stt": {
            "OPENAI_API_BASE_URL": request.app.state.config.STT_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.STT_OPENAI_API_KEY,
            "ENGINE": normalize_stt_engine(request.app.state.config.STT_ENGINE),
            "MODEL": request.app.state.config.STT_MODEL,
            "SUPPORTED_CONTENT_TYPES": request.app.state.config.STT_SUPPORTED_CONTENT_TYPES,
            "AZURE_API_KEY": request.app.state.config.AUDIO_STT_AZURE_API_KEY,
            "AZURE_REGION": request.app.state.config.AUDIO_STT_AZURE_REGION,
            "AZURE_LOCALES": request.app.state.config.AUDIO_STT_AZURE_LOCALES,
            "AZURE_BASE_URL": request.app.state.config.AUDIO_STT_AZURE_BASE_URL,
            "AZURE_MAX_SPEAKERS": request.app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS,
        },
    }


@router.post("/config/update")
async def update_audio_config(
    request: Request, form_data: AudioConfigUpdateForm, user=Depends(get_admin_user)
):
    request.app.state.config.TTS_OPENAI_API_BASE_URL = form_data.tts.OPENAI_API_BASE_URL
    request.app.state.config.TTS_OPENAI_API_KEY = form_data.tts.OPENAI_API_KEY
    request.app.state.config.TTS_OPENAI_PARAMS = form_data.tts.OPENAI_PARAMS
    request.app.state.config.TTS_GEMINI_API_BASE_URL = form_data.tts.GEMINI_API_BASE_URL
    request.app.state.config.TTS_GEMINI_API_KEY = form_data.tts.GEMINI_API_KEY
    request.app.state.config.TTS_GEMINI_PARAMS = form_data.tts.GEMINI_PARAMS
    request.app.state.config.TTS_GEMINI_SCENE = form_data.tts.GEMINI_SCENE
    request.app.state.config.TTS_GEMINI_SAMPLE_CONTEXT = form_data.tts.GEMINI_SAMPLE_CONTEXT
    request.app.state.config.TTS_GEMINI_STYLE = form_data.tts.GEMINI_STYLE
    request.app.state.config.TTS_GEMINI_PACE = form_data.tts.GEMINI_PACE
    request.app.state.config.TTS_GEMINI_ACCENT = form_data.tts.GEMINI_ACCENT
    request.app.state.config.TTS_GEMINI_TEMPERATURE = form_data.tts.GEMINI_TEMPERATURE
    request.app.state.config.TTS_API_KEY = form_data.tts.API_KEY
    request.app.state.config.TTS_ENGINE = form_data.tts.ENGINE
    request.app.state.config.TTS_MODEL = form_data.tts.MODEL
    request.app.state.config.TTS_VOICE = form_data.tts.VOICE
    request.app.state.config.TTS_SPLIT_ON = form_data.tts.SPLIT_ON
    request.app.state.config.TTS_AZURE_SPEECH_REGION = form_data.tts.AZURE_SPEECH_REGION
    request.app.state.config.TTS_AZURE_SPEECH_BASE_URL = (
        form_data.tts.AZURE_SPEECH_BASE_URL
    )
    request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT = (
        form_data.tts.AZURE_SPEECH_OUTPUT_FORMAT
    )

    request.app.state.config.STT_OPENAI_API_BASE_URL = form_data.stt.OPENAI_API_BASE_URL
    request.app.state.config.STT_OPENAI_API_KEY = form_data.stt.OPENAI_API_KEY
    request.app.state.config.STT_ENGINE = normalize_stt_engine(form_data.stt.ENGINE)
    request.app.state.config.STT_MODEL = form_data.stt.MODEL
    request.app.state.config.STT_SUPPORTED_CONTENT_TYPES = (
        form_data.stt.SUPPORTED_CONTENT_TYPES
    )

    request.app.state.config.AUDIO_STT_AZURE_API_KEY = form_data.stt.AZURE_API_KEY
    request.app.state.config.AUDIO_STT_AZURE_REGION = form_data.stt.AZURE_REGION
    request.app.state.config.AUDIO_STT_AZURE_LOCALES = form_data.stt.AZURE_LOCALES
    request.app.state.config.AUDIO_STT_AZURE_BASE_URL = form_data.stt.AZURE_BASE_URL
    request.app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS = (
        form_data.stt.AZURE_MAX_SPEAKERS
    )

    return {
        "tts": {
            "ENGINE": request.app.state.config.TTS_ENGINE,
            "MODEL": request.app.state.config.TTS_MODEL,
            "VOICE": request.app.state.config.TTS_VOICE,
            "OPENAI_API_BASE_URL": request.app.state.config.TTS_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.TTS_OPENAI_API_KEY,
            "OPENAI_PARAMS": request.app.state.config.TTS_OPENAI_PARAMS,
            "GEMINI_API_BASE_URL": request.app.state.config.TTS_GEMINI_API_BASE_URL,
            "GEMINI_API_KEY": request.app.state.config.TTS_GEMINI_API_KEY,
            "GEMINI_PARAMS": request.app.state.config.TTS_GEMINI_PARAMS,
            "GEMINI_SCENE": request.app.state.config.TTS_GEMINI_SCENE,
            "GEMINI_SAMPLE_CONTEXT": request.app.state.config.TTS_GEMINI_SAMPLE_CONTEXT,
            "GEMINI_STYLE": request.app.state.config.TTS_GEMINI_STYLE,
            "GEMINI_PACE": request.app.state.config.TTS_GEMINI_PACE,
            "GEMINI_ACCENT": request.app.state.config.TTS_GEMINI_ACCENT,
            "GEMINI_TEMPERATURE": request.app.state.config.TTS_GEMINI_TEMPERATURE,
            "API_KEY": request.app.state.config.TTS_API_KEY,
            "SPLIT_ON": request.app.state.config.TTS_SPLIT_ON,
            "AZURE_SPEECH_REGION": request.app.state.config.TTS_AZURE_SPEECH_REGION,
            "AZURE_SPEECH_BASE_URL": request.app.state.config.TTS_AZURE_SPEECH_BASE_URL,
            "AZURE_SPEECH_OUTPUT_FORMAT": request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT,
        },
        "stt": {
            "OPENAI_API_BASE_URL": request.app.state.config.STT_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.STT_OPENAI_API_KEY,
            "ENGINE": normalize_stt_engine(request.app.state.config.STT_ENGINE),
            "MODEL": request.app.state.config.STT_MODEL,
            "SUPPORTED_CONTENT_TYPES": request.app.state.config.STT_SUPPORTED_CONTENT_TYPES,
            "AZURE_API_KEY": request.app.state.config.AUDIO_STT_AZURE_API_KEY,
            "AZURE_REGION": request.app.state.config.AUDIO_STT_AZURE_REGION,
            "AZURE_LOCALES": request.app.state.config.AUDIO_STT_AZURE_LOCALES,
            "AZURE_BASE_URL": request.app.state.config.AUDIO_STT_AZURE_BASE_URL,
            "AZURE_MAX_SPEAKERS": request.app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS,
        },
    }


@router.post("/speech")
async def speech(request: Request, user=Depends(get_verified_user)):
    body = await request.body()
    cache_config = {
        "engine": request.app.state.config.TTS_ENGINE,
        "model": request.app.state.config.TTS_MODEL,
    }

    if request.app.state.config.TTS_ENGINE == "gemini":
        cache_config.update(
            {
                "voice": request.app.state.config.TTS_VOICE,
                "api_base_url": request.app.state.config.TTS_GEMINI_API_BASE_URL,
                "params": request.app.state.config.TTS_GEMINI_PARAMS,
                "scene": request.app.state.config.TTS_GEMINI_SCENE,
                "sample_context": request.app.state.config.TTS_GEMINI_SAMPLE_CONTEXT,
                "style": request.app.state.config.TTS_GEMINI_STYLE,
                "pace": request.app.state.config.TTS_GEMINI_PACE,
                "accent": request.app.state.config.TTS_GEMINI_ACCENT,
                "temperature": request.app.state.config.TTS_GEMINI_TEMPERATURE,
            }
        )

    name = hashlib.sha256(
        body + json.dumps(cache_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    file_extension = (
        "wav" if request.app.state.config.TTS_ENGINE == "gemini" else "mp3"
    )
    file_path = SPEECH_CACHE_DIR.joinpath(f"{name}.{file_extension}")
    file_body_path = SPEECH_CACHE_DIR.joinpath(f"{name}.json")

    # Check if the file already exists in the cache
    if file_path.is_file():
        return FileResponse(file_path)

    payload = None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as e:
        log.exception(e)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    r = None
    if request.app.state.config.TTS_ENGINE == "openai":
        payload["model"] = request.app.state.config.TTS_MODEL

        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                payload = {
                    **payload,
                    **(request.app.state.config.TTS_OPENAI_PARAMS or {}),
                }

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}",
                }
                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                r = await session.post(
                    url=f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/speech",
                    json=payload,
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                )

                r.raise_for_status()

                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(await r.read())

                async with aiofiles.open(file_body_path, "w") as f:
                    await f.write(json.dumps(payload))

            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)
            detail = None

            status_code = 500
            detail = f"Open WebUI: Server Connection Error"

            if r is not None:
                status_code = r.status

                try:
                    res = await r.json()
                    if "error" in res:
                        detail = f"External: {res['error']}"
                except Exception:
                    detail = f"External: {e}"

            raise HTTPException(
                status_code=status_code,
                detail=detail,
            )

    elif request.app.state.config.TTS_ENGINE == "gemini":
        api_key = request.app.state.config.TTS_GEMINI_API_KEY
        if not api_key:
            raise HTTPException(status_code=400, detail="Gemini API key is required")

        model = request.app.state.config.TTS_MODEL or "gemini-3.1-flash-tts-preview"
        voice_name = (
            payload.get("voice") or request.app.state.config.TTS_VOICE or "Zephyr"
        )
        transcript = payload.get("input", "")

        generation_config = {
            **(request.app.state.config.TTS_GEMINI_PARAMS or {}),
            "responseModalities": ["AUDIO"],
            "temperature": request.app.state.config.TTS_GEMINI_TEMPERATURE,
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name,
                    }
                }
            },
        }

        gemini_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": build_gemini_tts_prompt(
                                request.app.state.config.TTS_GEMINI_SCENE,
                                request.app.state.config.TTS_GEMINI_SAMPLE_CONTEXT,
                                transcript,
                                request.app.state.config.TTS_GEMINI_STYLE,
                                request.app.state.config.TTS_GEMINI_PACE,
                                request.app.state.config.TTS_GEMINI_ACCENT,
                            )
                        }
                    ],
                }
            ],
            "generationConfig": generation_config,
        }

        base_url = (
            request.app.state.config.TTS_GEMINI_API_BASE_URL
            or "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        if not base_url.endswith("/v1beta"):
            base_url = f"{base_url}/v1beta"

        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                headers = {"Content-Type": "application/json"}
                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                async with session.post(
                    url=f"{base_url}/models/{model}:generateContent",
                    params={"key": api_key},
                    json=gemini_payload,
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    response_text = await r.text()

                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_data = None

                    if r.status >= 400:
                        detail = response_text
                        if isinstance(response_data, dict) and "error" in response_data:
                            error = response_data["error"]
                            detail = error.get("message", error)
                        raise HTTPException(
                            status_code=r.status, detail=f"External: {detail}"
                        )

                    if response_data is None:
                        raise ValueError("Invalid JSON response from Gemini TTS")

                    audio_data, mime_type = extract_gemini_audio(response_data)
                    if not audio_data or not mime_type:
                        raise ValueError("No audio data returned from Gemini TTS")

                    audio_data, _, media_type = prepare_gemini_audio_file(
                        audio_data, mime_type
                    )

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(audio_data)

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(gemini_payload))

                    return FileResponse(file_path, media_type=media_type)

        except HTTPException:
            raise
        except Exception as e:
            log.exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Open WebUI: Server Connection Error",
            )

    elif request.app.state.config.TTS_ENGINE == "azure":
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            log.exception(e)
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        region = request.app.state.config.TTS_AZURE_SPEECH_REGION or "eastus"
        base_url = request.app.state.config.TTS_AZURE_SPEECH_BASE_URL
        language = request.app.state.config.TTS_VOICE
        locale = "-".join(request.app.state.config.TTS_VOICE.split("-")[:1])
        output_format = request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT

        try:
            data = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{locale}">
                <voice name="{language}">{html.escape(payload["input"])}</voice>
            </speak>"""
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                async with session.post(
                    (base_url or f"https://{region}.tts.speech.microsoft.com")
                    + "/cognitiveservices/v1",
                    headers={
                        "Ocp-Apim-Subscription-Key": request.app.state.config.TTS_API_KEY,
                        "Content-Type": "application/ssml+xml",
                        "X-Microsoft-OutputFormat": output_format,
                    },
                    data=data,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    r.raise_for_status()

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await r.read())

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(payload))

                    return FileResponse(file_path)

        except Exception as e:
            log.exception(e)
            detail = None

            try:
                if r.status != 200:
                    res = await r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )


def transcription_handler(request, file_path, metadata, user=None):
    filename = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    id = filename.split(".")[0]

    metadata = metadata or {}

    languages = [
        metadata.get("language", None),
        None,  # Always fallback to None in case transcription fails
    ]

    stt_engine = normalize_stt_engine(request.app.state.config.STT_ENGINE)

    if stt_engine == "openai":
        r = None
        try:
            for language in languages:
                payload = {
                    "model": request.app.state.config.STT_MODEL,
                }

                if language:
                    payload["language"] = language

                headers = {
                    "Authorization": f"Bearer {request.app.state.config.STT_OPENAI_API_KEY}"
                }
                if user and ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                with open(file_path, "rb") as audio_file:
                    r = requests.post(
                        url=f"{request.app.state.config.STT_OPENAI_API_BASE_URL}/audio/transcriptions",
                        headers=headers,
                        files={"file": (filename, audio_file)},
                        data=payload,
                    )

                if r.status_code == 200:
                    break

            r.raise_for_status()
            data = r.json()

            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            return data
        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
                except Exception:
                    detail = f"External: {e}"

            raise Exception(detail if detail else "Open WebUI: Server Connection Error")

    elif stt_engine == "azure":
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Audio file not found")

        file_size = os.path.getsize(file_path)
        if file_size > AZURE_MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds Azure limit of {AZURE_MAX_FILE_SIZE_MB}MB",
            )

        api_key = request.app.state.config.AUDIO_STT_AZURE_API_KEY
        region = request.app.state.config.AUDIO_STT_AZURE_REGION or "eastus"
        locales = request.app.state.config.AUDIO_STT_AZURE_LOCALES
        base_url = request.app.state.config.AUDIO_STT_AZURE_BASE_URL
        max_speakers = request.app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS or 3

        if len(locales) < 2:
            locales = [
                "en-US",
                "es-ES",
                "es-MX",
                "fr-FR",
                "hi-IN",
                "it-IT",
                "de-DE",
                "en-GB",
                "en-IN",
                "ja-JP",
                "ko-KR",
                "pt-BR",
                "zh-CN",
            ]
            locales = ",".join(locales)

        if not api_key or not region:
            raise HTTPException(
                status_code=400,
                detail="Azure API key is required for Azure STT",
            )

        r = None
        try:
            data = {
                "definition": json.dumps(
                    {
                        "locales": locales.split(","),
                        "diarization": {"maxSpeakers": max_speakers, "enabled": True},
                    }
                    if locales
                    else {}
                )
            }

            url = (
                base_url or f"https://{region}.api.cognitive.microsoft.com"
            ) + "/speechtotext/transcriptions:transcribe?api-version=2024-11-15"

            with open(file_path, "rb") as audio_file:
                r = requests.post(
                    url=url,
                    files={"audio": audio_file},
                    data=data,
                    headers={
                        "Ocp-Apim-Subscription-Key": api_key,
                    },
                )

            r.raise_for_status()
            response = r.json()

            if not response.get("combinedPhrases"):
                raise ValueError("No transcription found in response")

            transcript = response["combinedPhrases"][0].get("text", "").strip()
            if not transcript:
                raise ValueError("Empty transcript in response")

            data = {"text": transcript}

            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            log.debug(data)
            return data

        except (KeyError, IndexError, ValueError) as e:
            log.exception("Error parsing Azure response")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse Azure response: {str(e)}",
            )
        except requests.exceptions.RequestException as e:
            log.exception(e)
            detail = None

            try:
                if r is not None and r.status_code != 200:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status_code", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

    elif stt_engine == "web":
        raise HTTPException(
            status_code=400,
            detail="Web STT is browser-only and does not support server-side transcription",
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported STT engine: {stt_engine}",
    )

def transcribe(
    request: Request, file_path: str, metadata: Optional[dict] = None, user=None
):
    log.info(f"transcribe: {file_path} {metadata}")

    if is_audio_conversion_required(file_path):
        file_path = convert_audio_to_mp3(file_path)

    try:
        file_path = compress_audio(file_path)
    except Exception as e:
        log.exception(e)

    # Always produce a list of chunk paths (could be one entry if small)
    try:
        chunk_paths = split_audio(file_path, MAX_FILE_SIZE)
        print(f"Chunk paths: {chunk_paths}")
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )

    results = []
    try:
        with ThreadPoolExecutor() as executor:
            # Submit tasks for each chunk_path
            futures = [
                executor.submit(
                    transcription_handler, request, chunk_path, metadata, user
                )
                for chunk_path in chunk_paths
            ]
            # Gather results as they complete
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as transcribe_exc:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error transcribing chunk: {transcribe_exc}",
                    )
    finally:
        # Clean up only the temporary chunks, never the original file
        for chunk_path in chunk_paths:
            if chunk_path != file_path and os.path.isfile(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass

    return {
        "text": " ".join([result["text"] for result in results]),
    }


def compress_audio(file_path):
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        id = os.path.splitext(os.path.basename(file_path))[
            0
        ]  # Handles names with multiple dots
        file_dir = os.path.dirname(file_path)

        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Compress audio

        compressed_path = os.path.join(file_dir, f"{id}_compressed.mp3")
        audio.export(compressed_path, format="mp3", bitrate="32k")
        # log.debug(f"Compressed audio to {compressed_path}")  # Uncomment if log is defined

        return compressed_path
    else:
        return file_path


def split_audio(file_path, max_bytes, format="mp3", bitrate="32k"):
    """
    Splits audio into chunks not exceeding max_bytes.
    Returns a list of chunk file paths. If audio fits, returns list with original path.
    """
    file_size = os.path.getsize(file_path)
    if file_size <= max_bytes:
        return [file_path]  # Nothing to split

    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    orig_size = file_size

    approx_chunk_ms = max(int(duration_ms * (max_bytes / orig_size)) - 1000, 1000)
    chunks = []
    start = 0
    i = 0

    base, _ = os.path.splitext(file_path)

    while start < duration_ms:
        end = min(start + approx_chunk_ms, duration_ms)
        chunk = audio[start:end]
        chunk_path = f"{base}_chunk_{i}.{format}"
        chunk.export(chunk_path, format=format, bitrate=bitrate)

        # Reduce chunk duration if still too large
        while os.path.getsize(chunk_path) > max_bytes and (end - start) > 5000:
            end = start + ((end - start) // 2)
            chunk = audio[start:end]
            chunk.export(chunk_path, format=format, bitrate=bitrate)

        if os.path.getsize(chunk_path) > max_bytes:
            os.remove(chunk_path)
            raise Exception("Audio chunk cannot be reduced below max file size.")

        chunks.append(chunk_path)
        start = end
        i += 1

    return chunks


@router.post("/transcriptions")
def transcription(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    user=Depends(get_verified_user),
):
    log.info(f"file.content_type: {file.content_type}")
    stt_supported_content_types = getattr(
        request.app.state.config, "STT_SUPPORTED_CONTENT_TYPES", []
    )

    if not strict_match_mime_type(stt_supported_content_types, file.content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_NOT_SUPPORTED,
        )

    try:
        ext = file.filename.split(".")[-1]
        id = uuid.uuid4()

        filename = f"{id}.{ext}"
        contents = file.file.read()

        file_dir = f"{CACHE_DIR}/audio/transcriptions"
        os.makedirs(file_dir, exist_ok=True)
        file_path = f"{file_dir}/{filename}"

        with open(file_path, "wb") as f:
            f.write(contents)

        try:
            metadata = None

            if language:
                metadata = {"language": language}

            result = transcribe(request, file_path, metadata, user)

            return {
                **result,
                "filename": os.path.basename(file_path),
            }

        except Exception as e:
            log.exception(e)

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )

    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def get_available_models(request: Request) -> list[dict]:
    available_models = []
    if request.app.state.config.TTS_ENGINE == "openai":
        # Use custom endpoint if not using the official OpenAI API URL
        if not request.app.state.config.TTS_OPENAI_API_BASE_URL.startswith(
            "https://api.openai.com"
        ):
            try:
                response = requests.get(
                    f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/models"
                )
                response.raise_for_status()
                data = response.json()
                available_models = data.get("models", [])
            except Exception as e:
                log.error(f"Error fetching models from custom endpoint: {str(e)}")
                available_models = [{"id": "tts-1"}, {"id": "tts-1-hd"}]
        else:
            available_models = [{"id": "tts-1"}, {"id": "tts-1-hd"}]
    elif request.app.state.config.TTS_ENGINE == "gemini":
        available_models = GEMINI_TTS_MODELS
    return available_models


@router.get("/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    return {"models": get_available_models(request)}


def get_available_voices(request) -> dict:
    """Returns {voice_id: voice_name} dict"""
    available_voices = {}
    if request.app.state.config.TTS_ENGINE == "openai":
        # Use custom endpoint if not using the official OpenAI API URL
        if not request.app.state.config.TTS_OPENAI_API_BASE_URL.startswith(
            "https://api.openai.com"
        ):
            try:
                response = requests.get(
                    f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/voices"
                )
                response.raise_for_status()
                data = response.json()
                voices_list = data.get("voices", [])
                available_voices = {voice["id"]: voice["name"] for voice in voices_list}
            except Exception as e:
                log.error(f"Error fetching voices from custom endpoint: {str(e)}")
                available_voices = {
                    "alloy": "alloy",
                    "echo": "echo",
                    "fable": "fable",
                    "onyx": "onyx",
                    "nova": "nova",
                    "shimmer": "shimmer",
                }
        else:
            available_voices = {
                "alloy": "alloy",
                "echo": "echo",
                "fable": "fable",
                "onyx": "onyx",
                "nova": "nova",
                "shimmer": "shimmer",
            }
    elif request.app.state.config.TTS_ENGINE == "gemini":
        available_voices = GEMINI_TTS_VOICES
    elif request.app.state.config.TTS_ENGINE == "azure":
        try:
            region = request.app.state.config.TTS_AZURE_SPEECH_REGION
            base_url = request.app.state.config.TTS_AZURE_SPEECH_BASE_URL
            url = (
                base_url or f"https://{region}.tts.speech.microsoft.com"
            ) + "/cognitiveservices/voices/list"
            headers = {
                "Ocp-Apim-Subscription-Key": request.app.state.config.TTS_API_KEY
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            voices = response.json()

            for voice in voices:
                available_voices[voice["ShortName"]] = (
                    f"{voice['DisplayName']} ({voice['ShortName']})"
                )
        except requests.RequestException as e:
            log.error(f"Error fetching voices: {str(e)}")

    return available_voices


@router.get("/voices")
async def get_voices(request: Request, user=Depends(get_verified_user)):
    return {
        "voices": [
            {"id": k, "name": v} for k, v in get_available_voices(request).items()
        ]
    }
