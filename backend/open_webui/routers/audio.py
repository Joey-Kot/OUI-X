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
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask


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

OPENAI_TTS_MODELS = [
    {"id": "gpt-4o-mini-tts"},
    {"id": "tts-1"},
    {"id": "tts-1-hd"},
]

GEMINI_TTS_MODELS = [
    {"id": "gemini-3.1-flash-tts-preview"},
    {"id": "gemini-2.5-flash-preview-tts"},
    {"id": "gemini-2.5-pro-preview-tts"},
]

QWEN_TTS_MODELS = [
    {"id": "qwen3-tts-flash"},
    {"id": "qwen3-tts-instruct-flash"},
]

OPENAI_TTS_VOICES = {
    "alloy": "alloy",
    "ash": "ash",
    "ballad": "ballad",
    "coral": "coral",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "sage": "sage",
    "shimmer": "shimmer",
    "verse": "verse",
    "marin": "marin",
    "cedar": "cedar",
}

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

QWEN_TTS_VOICES = {
    "Cherry": "Cherry - Sunny, positive, friendly female voice",
    "Serena": "Serena - Gentle female voice",
    "Ethan": "Ethan - Warm, energetic male voice",
    "Chelsie": "Chelsie - Anime-style female voice",
    "Momo": "Momo - Playful female voice",
    "Vivian": "Vivian - Cute, spirited female voice",
    "Moon": "Moon - Free-spirited male voice",
    "Maia": "Maia - Intellectual and gentle female voice",
    "Kai": "Kai - Relaxed male voice",
    "Nofish": "Nofish - Designer-style male voice",
    "Bella": "Bella - Little-girl female voice",
    "Jennifer": "Jennifer - Cinematic American female voice",
    "Ryan": "Ryan - Dramatic, rhythmic male voice",
    "Katerina": "Katerina - Mature female voice",
    "Aiden": "Aiden - American young male voice",
    "Eldric Sage": "Eldric Sage - Calm, wise elder male voice",
    "Mia": "Mia - Sweet and obedient female voice",
    "Mochi": "Mochi - Clever childlike male voice",
    "Bellona": "Bellona - Loud, clear, vivid female voice",
    "Vincent": "Vincent - Hoarse, rugged male voice",
    "Bunny": "Bunny - Cute little-girl female voice",
    "Neil": "Neil - Professional news anchor male voice",
    "Elias": "Elias - Knowledge narrator female voice",
    "Arthur": "Arthur - Rustic elder male voice",
    "Nini": "Nini - Soft, sweet girl-next-door female voice",
    "Seren": "Seren - Calm sleep-aid female voice",
    "Pip": "Pip - Mischievous childlike male voice",
    "Stella": "Stella - Sweet magical-girl female voice",
    "Bodega": "Bodega - Warm Spanish male voice",
    "Sonrisa": "Sonrisa - Cheerful Latin female voice",
    "Alek": "Alek - Russian-accented male voice",
    "Dolce": "Dolce - Lazy Italian male voice",
    "Sohee": "Sohee - Expressive Korean female voice",
    "Ono Anna": "Ono Anna - Playful Japanese female voice",
    "Lenn": "Lenn - German young male voice",
    "Emilien": "Emilien - Romantic French male voice",
    "Andre": "Andre - Magnetic, steady male voice",
    "Radio Gol": "Radio Gol - Football commentator male voice",
    "Jada": "Jada - Shanghainese female voice",
    "Dylan": "Dylan - Beijing male voice",
    "Li": "Li - Nanjing male voice",
    "Marcus": "Marcus - Shaanxi male voice",
    "Roy": "Roy - Minnan male voice",
    "Peter": "Peter - Tianjin crosstalk male voice",
    "Sunny": "Sunny - Sichuan female voice",
    "Eric": "Eric - Sichuan male voice",
    "Rocky": "Rocky - Cantonese male voice",
    "Kiki": "Kiki - Cantonese female voice",
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

OPENAI_TTS_RESPONSE_FORMATS = {
    "mp3": ("mp3", "audio/mpeg"),
    "opus": ("opus", "audio/opus"),
    "aac": ("aac", "audio/aac"),
    "flac": ("flac", "audio/flac"),
    "wav": ("wav", "audio/wav"),
    "pcm": ("pcm", "audio/pcm"),
}

TTS_OUTPUT_FORMATS = {
    "webm": ("webm", "audio/webm", "webm", "libopus"),
    "mp3": ("mp3", "audio/mpeg", "mp3", None),
    "flac": ("flac", "audio/flac", "flac", None),
    "wav": ("wav", "audio/wav", "wav", None),
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
    return build_wav_header(mime_type, len(audio_data)) + audio_data


def build_wav_header(mime_type: str, data_size: Optional[int] = None) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = data_size if data_size is not None else 0xFFFFFFFF
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size if data_size <= 0xFFFFFFFF - 36 else 0xFFFFFFFF

    return struct.pack(
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


async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    if response:
        response.close()
    if session:
        await session.close()


async def iter_sse_json(stream: aiohttp.StreamReader):
    buffer = ""
    data_lines = []

    async for chunk, _ in stream.iter_chunks():
        if not chunk:
            continue

        buffer += chunk.decode("utf-8", errors="ignore")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")

            if not line:
                if data_lines:
                    data = "\n".join(data_lines).strip()
                    data_lines = []
                    if data and data != "[DONE]":
                        yield json.loads(data)
                continue

            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

    if data_lines:
        data = "\n".join(data_lines).strip()
        if data and data != "[DONE]":
            yield json.loads(data)


async def stream_raw_audio_chunks(stream: aiohttp.StreamReader):
    async for chunk, _ in stream.iter_chunks():
        if chunk:
            yield chunk


async def stream_gemini_audio_chunks(stream: aiohttp.StreamReader):
    yield build_wav_header("audio/L16;rate=24000")

    async for response_data in iter_sse_json(stream):
        audio_data, _ = extract_gemini_audio(response_data)
        if audio_data:
            yield audio_data


async def stream_qwen_audio_chunks(stream: aiohttp.StreamReader):
    sent_header = False

    async for response_data in iter_sse_json(stream):
        status_code = response_data.get("status_code")
        if status_code and status_code >= 400:
            raise ValueError(response_data.get("message") or "Qwen TTS stream error")

        audio_data, _ = extract_qwen_audio(response_data)
        if not audio_data:
            continue

        if not sent_header:
            sent_header = True
            if not audio_data.startswith(b"RIFF"):
                yield build_wav_header("audio/L16;rate=24000")

        yield audio_data


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


def build_openai_tts_url(api_base_url: str) -> str:
    base_url = (api_base_url or "https://api.openai.com/v1").rstrip("/")
    speech_path = "/audio/speech"

    if base_url.endswith(speech_path):
        return base_url

    if base_url == "https://api.openai.com":
        base_url = f"{base_url}/v1"

    return f"{base_url}{speech_path}"


def parse_openai_tts_voice(voice):
    if not isinstance(voice, str):
        return voice

    voice = voice.strip()
    if voice.startswith("{"):
        try:
            voice_data = json.loads(voice)
            if isinstance(voice_data, dict):
                return voice_data
        except json.JSONDecodeError:
            pass

    return voice


def get_openai_tts_response_format(params: Optional[dict]) -> tuple[str, str]:
    response_format = str((params or {}).get("response_format") or "mp3").lower()
    return OPENAI_TTS_RESPONSE_FORMATS.get(
        response_format, OPENAI_TTS_RESPONSE_FORMATS["mp3"]
    )


def normalize_tts_output_format(output_format: Optional[str]) -> str:
    output_format = (output_format or "default").strip().lower()
    if output_format == "default" or output_format in TTS_OUTPUT_FORMATS:
        return output_format
    return "default"


def get_tts_output_format_media_type(output_format: Optional[str]) -> Optional[str]:
    output_format = normalize_tts_output_format(output_format)
    if output_format == "default":
        return None
    return TTS_OUTPUT_FORMATS[output_format][1]


def get_tts_output_file_extension(
    source_extension: str, output_format: Optional[str]
) -> str:
    output_format = normalize_tts_output_format(output_format)
    if output_format == "default":
        return source_extension.lstrip(".") or "mp3"
    return TTS_OUTPUT_FORMATS[output_format][0]


async def save_tts_audio_file(
    audio_data: bytes,
    name: str,
    source_extension: str,
    source_media_type: Optional[str],
    output_format: Optional[str],
    cache_extension: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    output_format = normalize_tts_output_format(output_format)
    source_extension = (source_extension or "mp3").lstrip(".")

    if output_format == "default":
        file_extension = (cache_extension or source_extension).lstrip(".")
        file_path = SPEECH_CACHE_DIR.joinpath(f"{name}.{file_extension}")
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(audio_data)
        return str(file_path), source_media_type

    target_extension, target_media_type, export_format, codec = TTS_OUTPUT_FORMATS[
        output_format
    ]
    source_path = SPEECH_CACHE_DIR.joinpath(f"{name}.source.{source_extension}")
    target_path = SPEECH_CACHE_DIR.joinpath(f"{name}.{target_extension}")

    async with aiofiles.open(source_path, "wb") as f:
        await f.write(audio_data)

    audio = AudioSegment.from_file(source_path)
    export_kwargs = {}
    if codec:
        export_kwargs["codec"] = codec
    audio.export(target_path, format=export_format, **export_kwargs)
    try:
        source_path.unlink(missing_ok=True)
    except OSError:
        log.debug("Failed to remove temporary TTS source file: %s", source_path)

    return str(target_path), target_media_type


def build_qwen_tts_url(api_base_url: str) -> str:
    base_url = (api_base_url or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
    generation_path = "/services/aigc/multimodal-generation/generation"

    if base_url.endswith(generation_path):
        return base_url

    if not base_url.endswith("/api/v1"):
        base_url = f"{base_url}/api/v1"

    return f"{base_url}{generation_path}"


def extract_qwen_audio(response_data):
    audio = response_data.get("output", {}).get("audio", {})
    if not audio:
        return None, None

    if audio.get("data"):
        return base64.b64decode(audio["data"]), None

    return None, audio.get("url")


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
    QWEN_API_BASE_URL: str = "https://dashscope.aliyuncs.com/api/v1"
    QWEN_API_KEY: str = ""
    QWEN_PARAMS: Optional[dict] = None
    API_KEY: str
    ENGINE: str
    MODEL: str
    VOICE: str
    SPLIT_ON: str
    STREAM_RESPONSE: bool = False
    OUTPUT_FORMAT: str = "default"
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
            "QWEN_API_BASE_URL": request.app.state.config.TTS_QWEN_API_BASE_URL,
            "QWEN_API_KEY": request.app.state.config.TTS_QWEN_API_KEY,
            "QWEN_PARAMS": request.app.state.config.TTS_QWEN_PARAMS,
            "API_KEY": request.app.state.config.TTS_API_KEY,
            "ENGINE": request.app.state.config.TTS_ENGINE,
            "MODEL": request.app.state.config.TTS_MODEL,
            "VOICE": request.app.state.config.TTS_VOICE,
            "SPLIT_ON": request.app.state.config.TTS_SPLIT_ON,
            "STREAM_RESPONSE": request.app.state.config.TTS_STREAM_RESPONSE,
            "OUTPUT_FORMAT": request.app.state.config.TTS_OUTPUT_FORMAT,
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
    request.app.state.config.TTS_QWEN_API_BASE_URL = form_data.tts.QWEN_API_BASE_URL
    request.app.state.config.TTS_QWEN_API_KEY = form_data.tts.QWEN_API_KEY
    request.app.state.config.TTS_QWEN_PARAMS = form_data.tts.QWEN_PARAMS
    request.app.state.config.TTS_API_KEY = form_data.tts.API_KEY
    request.app.state.config.TTS_ENGINE = form_data.tts.ENGINE
    request.app.state.config.TTS_MODEL = form_data.tts.MODEL
    request.app.state.config.TTS_VOICE = form_data.tts.VOICE
    request.app.state.config.TTS_SPLIT_ON = form_data.tts.SPLIT_ON
    request.app.state.config.TTS_STREAM_RESPONSE = form_data.tts.STREAM_RESPONSE
    request.app.state.config.TTS_OUTPUT_FORMAT = normalize_tts_output_format(
        form_data.tts.OUTPUT_FORMAT
    )
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
            "QWEN_API_BASE_URL": request.app.state.config.TTS_QWEN_API_BASE_URL,
            "QWEN_API_KEY": request.app.state.config.TTS_QWEN_API_KEY,
            "QWEN_PARAMS": request.app.state.config.TTS_QWEN_PARAMS,
            "API_KEY": request.app.state.config.TTS_API_KEY,
            "SPLIT_ON": request.app.state.config.TTS_SPLIT_ON,
            "STREAM_RESPONSE": request.app.state.config.TTS_STREAM_RESPONSE,
            "OUTPUT_FORMAT": request.app.state.config.TTS_OUTPUT_FORMAT,
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
    tts_output_format = normalize_tts_output_format(
        request.app.state.config.TTS_OUTPUT_FORMAT
    )
    tts_stream_response = bool(
        getattr(request.app.state.config, "TTS_STREAM_RESPONSE", False)
    ) and request.app.state.config.TTS_ENGINE in {"openai", "gemini", "qwen"}
    cache_config = {
        "engine": request.app.state.config.TTS_ENGINE,
        "model": request.app.state.config.TTS_MODEL,
        "output_format": tts_output_format,
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
    elif request.app.state.config.TTS_ENGINE == "openai":
        cache_config.update(
            {
                "voice": request.app.state.config.TTS_VOICE,
                "api_base_url": request.app.state.config.TTS_OPENAI_API_BASE_URL,
                "params": request.app.state.config.TTS_OPENAI_PARAMS,
            }
        )
    elif request.app.state.config.TTS_ENGINE == "qwen":
        cache_config.update(
            {
                "voice": request.app.state.config.TTS_VOICE,
                "api_base_url": request.app.state.config.TTS_QWEN_API_BASE_URL,
                "params": request.app.state.config.TTS_QWEN_PARAMS,
            }
        )

    name = hashlib.sha256(
        body + json.dumps(cache_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    file_extension = "mp3"
    if request.app.state.config.TTS_ENGINE == "openai":
        file_extension, _ = get_openai_tts_response_format(
            request.app.state.config.TTS_OPENAI_PARAMS
        )
    elif request.app.state.config.TTS_ENGINE in {"gemini", "qwen"}:
        file_extension = "wav"
    file_extension = get_tts_output_file_extension(file_extension, tts_output_format)
    file_path = SPEECH_CACHE_DIR.joinpath(f"{name}.{file_extension}")
    file_body_path = SPEECH_CACHE_DIR.joinpath(f"{name}.json")

    # Check if the file already exists in the cache
    if not tts_stream_response and file_path.is_file():
        return FileResponse(
            file_path, media_type=get_tts_output_format_media_type(tts_output_format)
        )

    payload = None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as e:
        log.exception(e)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if request.app.state.config.TTS_ENGINE == "openai":
        model = request.app.state.config.TTS_MODEL or "gpt-4o-mini-tts"
        voice = payload.get("voice") or request.app.state.config.TTS_VOICE or "alloy"
        transcript = payload.get("input", "")
        source_extension, default_media_type = get_openai_tts_response_format(
            request.app.state.config.TTS_OPENAI_PARAMS
        )
        openai_payload = {
            **(request.app.state.config.TTS_OPENAI_PARAMS or {}),
            "model": model,
            "input": transcript,
            "voice": parse_openai_tts_voice(voice),
        }

        if tts_stream_response:
            session = None
            r = None
            try:
                timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
                session = aiohttp.ClientSession(timeout=timeout, trust_env=True)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}",
                }
                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                r = await session.post(
                    url=build_openai_tts_url(
                        request.app.state.config.TTS_OPENAI_API_BASE_URL
                    ),
                    json=openai_payload,
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                )

                if r.status >= 400:
                    response_text = await r.text()
                    detail = response_text
                    try:
                        res = json.loads(response_text)
                        if "error" in res:
                            detail = res["error"]
                    except json.JSONDecodeError:
                        pass
                    raise HTTPException(
                        status_code=r.status, detail=f"External: {detail}"
                    )

                media_type = (
                    r.headers.get("Content-Type", "").split(";", 1)[0]
                    or default_media_type
                )
                return StreamingResponse(
                    stream_raw_audio_chunks(r.content),
                    media_type=media_type,
                    background=BackgroundTask(
                        cleanup_response, response=r, session=session
                    ),
                )
            except HTTPException:
                await cleanup_response(r, session)
                raise
            except Exception as e:
                await cleanup_response(r, session)
                log.exception(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Open WebUI: Server Connection Error",
                )

        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}",
                }
                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                async with session.post(
                    url=build_openai_tts_url(
                        request.app.state.config.TTS_OPENAI_API_BASE_URL
                    ),
                    json=openai_payload,
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    if r.status >= 400:
                        response_text = await r.text()
                        detail = response_text
                        try:
                            res = json.loads(response_text)
                            if "error" in res:
                                detail = res["error"]
                        except json.JSONDecodeError:
                            pass
                        raise HTTPException(
                            status_code=r.status, detail=f"External: {detail}"
                        )

                    audio_data = await r.read()
                    media_type = (
                        r.headers.get("Content-Type", "").split(";", 1)[0]
                        or default_media_type
                    )

                    final_path, final_media_type = await save_tts_audio_file(
                        audio_data,
                        name,
                        source_extension,
                        media_type,
                        tts_output_format,
                        file_extension,
                    )

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(openai_payload))

                    return FileResponse(final_path, media_type=final_media_type)

        except HTTPException:
            raise
        except Exception as e:
            log.exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Open WebUI: Server Connection Error",
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

        if tts_stream_response:
            session = None
            r = None
            try:
                timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
                session = aiohttp.ClientSession(timeout=timeout, trust_env=True)
                headers = {"Content-Type": "application/json"}
                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                r = await session.post(
                    url=f"{base_url}/models/{model}:streamGenerateContent",
                    params={"key": api_key, "alt": "sse"},
                    json=gemini_payload,
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                )

                if r.status >= 400:
                    response_text = await r.text()
                    detail = response_text
                    try:
                        res = json.loads(response_text)
                        if "error" in res:
                            error = res["error"]
                            detail = error.get("message", error)
                    except json.JSONDecodeError:
                        pass
                    raise HTTPException(
                        status_code=r.status, detail=f"External: {detail}"
                    )

                return StreamingResponse(
                    stream_gemini_audio_chunks(r.content),
                    media_type="audio/wav",
                    background=BackgroundTask(
                        cleanup_response, response=r, session=session
                    ),
                )
            except HTTPException:
                await cleanup_response(r, session)
                raise
            except Exception as e:
                await cleanup_response(r, session)
                log.exception(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Open WebUI: Server Connection Error",
                )

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

                    audio_data, source_extension, media_type = prepare_gemini_audio_file(
                        audio_data, mime_type
                    )

                    final_path, final_media_type = await save_tts_audio_file(
                        audio_data,
                        name,
                        source_extension,
                        media_type,
                        tts_output_format,
                        file_extension,
                    )

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(gemini_payload))

                    return FileResponse(final_path, media_type=final_media_type)

        except HTTPException:
            raise
        except Exception as e:
            log.exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Open WebUI: Server Connection Error",
            )

    elif request.app.state.config.TTS_ENGINE == "qwen":
        api_key = request.app.state.config.TTS_QWEN_API_KEY
        if not api_key:
            raise HTTPException(status_code=400, detail="Qwen API key is required")

        model = request.app.state.config.TTS_MODEL or "qwen3-tts-flash"
        voice = payload.get("voice") or request.app.state.config.TTS_VOICE or "Cherry"
        transcript = payload.get("input", "")

        qwen_payload = {
            "model": model,
            "input": {
                **(request.app.state.config.TTS_QWEN_PARAMS or {}),
                "text": transcript,
                "voice": voice,
            },
        }

        if tts_stream_response:
            session = None
            r = None
            try:
                timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
                session = aiohttp.ClientSession(timeout=timeout, trust_env=True)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "X-DashScope-SSE": "enable",
                }
                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                r = await session.post(
                    url=build_qwen_tts_url(
                        request.app.state.config.TTS_QWEN_API_BASE_URL
                    ),
                    json=qwen_payload,
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                )

                if r.status >= 400:
                    response_text = await r.text()
                    detail = response_text
                    try:
                        response_data = json.loads(response_text)
                        detail = response_data.get("message") or detail
                    except json.JSONDecodeError:
                        pass
                    raise HTTPException(
                        status_code=r.status, detail=f"External: {detail}"
                    )

                return StreamingResponse(
                    stream_qwen_audio_chunks(r.content),
                    media_type="audio/wav",
                    background=BackgroundTask(
                        cleanup_response, response=r, session=session
                    ),
                )
            except HTTPException:
                await cleanup_response(r, session)
                raise
            except Exception as e:
                await cleanup_response(r, session)
                log.exception(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Open WebUI: Server Connection Error",
                )

        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                async with session.post(
                    url=build_qwen_tts_url(
                        request.app.state.config.TTS_QWEN_API_BASE_URL
                    ),
                    json=qwen_payload,
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
                        if isinstance(response_data, dict):
                            detail = response_data.get("message") or detail
                        raise HTTPException(
                            status_code=r.status, detail=f"External: {detail}"
                        )

                    if response_data is None:
                        raise ValueError("Invalid JSON response from Qwen TTS")

                    status_code = response_data.get("status_code")
                    if status_code and status_code >= 400:
                        detail = response_data.get("message") or response_text
                        raise HTTPException(
                            status_code=status_code, detail=f"External: {detail}"
                        )

                    audio_data, audio_url = extract_qwen_audio(response_data)
                    media_type = "audio/wav"
                    source_extension = "wav"

                    if audio_url:
                        async with session.get(
                            audio_url, ssl=AIOHTTP_CLIENT_SESSION_SSL
                        ) as audio_response:
                            audio_response.raise_for_status()
                            audio_data = await audio_response.read()
                            content_type = audio_response.headers.get(
                                "Content-Type", ""
                            ).split(";", 1)[0]
                            media_type = content_type or media_type
                            if media_type in {
                                "application/octet-stream",
                                "binary/octet-stream",
                            }:
                                media_type = (
                                    mimetypes.guess_type(
                                        audio_url.split("?", 1)[0]
                                    )[0]
                                    or media_type
                                )
                            source_extension = (
                                mimetypes.guess_extension(media_type)
                                or os.path.splitext(audio_url.split("?", 1)[0])[1]
                                or ".wav"
                            )

                    if not audio_data:
                        raise ValueError("No audio data returned from Qwen TTS")

                    final_path, final_media_type = await save_tts_audio_file(
                        audio_data,
                        name,
                        source_extension,
                        media_type,
                        tts_output_format,
                        file_extension,
                    )

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(qwen_payload))

                    return FileResponse(final_path, media_type=final_media_type)

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

                    audio_data = await r.read()
                    media_type = r.headers.get("Content-Type", "").split(";", 1)[0]
                    source_extension = mimetypes.guess_extension(media_type or "")
                    if not source_extension:
                        source_extension = "mp3" if "mp3" in output_format else "wav"

                    final_path, final_media_type = await save_tts_audio_file(
                        audio_data,
                        name,
                        source_extension,
                        media_type or None,
                        tts_output_format,
                        file_extension,
                    )

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(payload))

                    return FileResponse(final_path, media_type=final_media_type)

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
                headers = {}
                if request.app.state.config.TTS_OPENAI_API_KEY:
                    headers["Authorization"] = (
                        f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}"
                    )
                response = requests.get(
                    f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/models",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                available_models = data.get("models", [])
            except Exception as e:
                log.error(f"Error fetching models from custom endpoint: {str(e)}")
                available_models = OPENAI_TTS_MODELS
        else:
            available_models = OPENAI_TTS_MODELS
    elif request.app.state.config.TTS_ENGINE == "gemini":
        available_models = GEMINI_TTS_MODELS
    elif request.app.state.config.TTS_ENGINE == "qwen":
        available_models = QWEN_TTS_MODELS
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
                headers = {}
                if request.app.state.config.TTS_OPENAI_API_KEY:
                    headers["Authorization"] = (
                        f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}"
                    )
                response = requests.get(
                    f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/voices",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                voices_list = data.get("voices", [])
                available_voices = {voice["id"]: voice["name"] for voice in voices_list}
            except Exception as e:
                log.error(f"Error fetching voices from custom endpoint: {str(e)}")
                available_voices = OPENAI_TTS_VOICES
        else:
            available_voices = OPENAI_TTS_VOICES
    elif request.app.state.config.TTS_ENGINE == "gemini":
        available_voices = GEMINI_TTS_VOICES
    elif request.app.state.config.TTS_ENGINE == "qwen":
        available_voices = QWEN_TTS_VOICES
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
