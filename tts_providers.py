"""
TTS Provider implementations for benchmarking
"""
import time
import asyncio
import aiohttp
import requests
import ssl
import os
import base64
import certifi
from xml.sax.saxutils import escape as _xml_escape
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import io
import json
import wave
from config import (
    get_api_key,
    TTS_PROVIDERS,
    get_falcon_api_url,
    falcon_auth_headers,
    falcon_synthesis_timeout,
    NORMALIZATION,
)

def humanize_provider_error(provider_id: str, status: int, error_text: str) -> str:
    """Map raw vendor errors to actionable messages for operators/raters."""
    body = (error_text or "").strip()
    lower = body.lower()
    if provider_id.startswith("elevenlabs") and (
        "detected_unusual_activity" in lower or "unusual activity" in lower
    ):
        return (
            "ElevenLabs free tier blocks cloud hosting (Streamlit Cloud / datacenter IPs). "
            "Upgrade to a paid ElevenLabs plan, or remove ELEVENLABS_API_KEY from this deployment."
        )
    if provider_id.startswith("elevenlabs") and status == 401:
        return "ElevenLabs API key invalid or revoked. Check ELEVENLABS_API_KEY in secrets."
    if body:
        return f"API Error {status}: {body[:200]}"
    return f"API Error {status}"


# Arena standard: request WAV @ 24 kHz mono from every API that supports it.
ARENA_SAMPLE_RATE: int = int(NORMALIZATION.get("sample_rate", 24000))
ARENA_CHANNELS: int = int(NORMALIZATION.get("channels", 1))
ARENA_FORMAT: str = "wav"


def pcm16_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = ARENA_SAMPLE_RATE,
    channels: int = ARENA_CHANNELS,
) -> bytes:
    """Wrap raw 16-bit little-endian PCM in a minimal WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _wav_meta(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {"format": "wav", "sample_rate": ARENA_SAMPLE_RATE, "channels": ARENA_CHANNELS}
    if extra:
        meta.update(extra)
    return meta

def get_ssl_context():
    """Create SSL context with proper certificate handling"""
    try:
        # Try to use certifi certificates first
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        return ssl_context
    except Exception:
        # Fallback to no verification if certifi fails
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context

def get_connector():
    """Create aiohttp connector with SSL handling"""
    ssl_context = get_ssl_context()
    return aiohttp.TCPConnector(ssl=ssl_context)

@dataclass
class TTSResult:
    """Result from TTS generation"""
    success: bool
    audio_data: Optional[bytes]
    latency_ms: float  # TTFB: ms from request send to first response body byte
    file_size_bytes: int
    error_message: Optional[str]
    metadata: Dict[str, Any]
    latency_1: float = 0.0  # Network latency (pure RTT) without TTS processing

@dataclass
class TTSRequest:
    """TTS generation request"""
    text: str
    voice: str
    provider: str
    model: Optional[str] = None
    speed: float = 1.0
    format: str = ARENA_FORMAT
    sample_rate: int = ARENA_SAMPLE_RATE

class TTSProvider(ABC):
    """Abstract base class for TTS providers"""
    
    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        self.config = TTS_PROVIDERS[provider_id]
        self.api_key = get_api_key(provider_id)
    
    @abstractmethod
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech from text"""
        pass
    
    @abstractmethod
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        pass
    
    def validate_request(self, request: TTSRequest) -> Tuple[bool, str]:
        """Validate TTS request"""
        if len(request.text) > self.config.max_chars:
            return False, f"Text exceeds maximum length of {self.config.max_chars} characters"
        
        if request.voice not in self.config.supported_voices:
            return False, f"Voice '{request.voice}' not supported. Available: {self.config.supported_voices}"
        
        return True, ""
    
    @staticmethod
    async def read_body_ttfb(response, send_time: float) -> Tuple[bytes, float]:
        """Read the full response body via chunked streaming and return
        (body_bytes, ttfb_ms).

        ttfb_ms is the true time-to-first-byte: ms from ``send_time`` (captured
        right before the request was sent) to the moment the first response
        body chunk arrives. For non-streaming endpoints the body arrives in one
        piece, so this naturally collapses to the full server-side latency.
        """
        ttfb_ms: Optional[float] = None
        chunks: List[bytes] = []
        async for chunk in response.content.iter_chunked(4096):
            if chunk and ttfb_ms is None:
                ttfb_ms = (time.time() - send_time) * 1000
            chunks.append(chunk)
        body = b"".join(chunks)
        if ttfb_ms is None:  # empty body
            ttfb_ms = (time.time() - send_time) * 1000
        return body, ttfb_ms

    async def measure_ping_latency(self) -> float:
        """Measure pure network latency (RTT) without TTS processing"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                # Send a minimal HEAD or OPTIONS request to measure pure network latency
                async with session.head(
                    self.config.base_url,
                    headers={"api-key": self.api_key} if "murf" in self.provider_id else {"Authorization": f"Token {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    return latency_ms
        except:
            # If HEAD doesn't work, fallback to minimal GET/POST
            try:
                start_time = time.time()
                async with aiohttp.ClientSession(connector=get_connector()) as session:
                    async with session.get(
                        self.config.base_url.replace("/v1/speech/", "/").replace("turbo-stream", "").replace("stream", "").rstrip("/"),
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        latency_ms = (time.time() - start_time) * 1000
                        return latency_ms
            except:
                return 0.0  # Return 0 if ping fails

class MurfFalconOct23TTSProvider(TTSProvider):
    """Murf Falcon Oct 23 TTS provider implementation (Global Stream Endpoint)"""
    
    def __init__(self):
        super().__init__("murf_falcon_oct23")
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Murf Falcon Oct 23 API (Global Stream)"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Murf Falcon Oct 23 API payload structure
        payload = {
            "text": request.text,
            "voiceId": request.voice,
            "format": "mp3",
            "sampleRate": 24000,
            "model": "FALCON",
            "channelType": "MONO"
        }
        
        # Add speed/rate if specified
        if request.speed and request.speed != 1.0:
            payload["rate"] = request.speed
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        audio_data = await response.read()
                        file_size = len(audio_data)
                        
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=file_size,
                            error_message=None,
                            metadata={
                                "provider": self.provider_id,
                                "model": "FALCON",
                                "voice": request.voice,
                                "format": request.format or "mp3"
                            }
                        )
                    else:
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text}",
                            metadata={"provider": self.provider_id}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available Murf Falcon Oct 23 voices"""
        return self.config.supported_voices

class MurfGen2TTSProvider(TTSProvider):
    """Murf Speech Gen 2 (production streaming API, model=GEN2)"""

    def __init__(self):
        super().__init__("murf_gen2")

    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        start_time = time.time()
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={},
            )
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": request.text,
            "voiceId": request.voice,
            "format": "WAV",
            "sampleRate": ARENA_SAMPLE_RATE,
            "model": "GEN2",
            "channelType": "MONO",
        }
        if request.speed and request.speed != 1.0:
            payload["rate"] = request.speed

        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        audio_data, latency_ms = await self.read_body_ttfb(response, send_time)
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata=_wav_meta({
                                "provider": self.provider_id,
                                "model": "GEN2",
                                "voice": request.voice,
                            }),
                        )
                    latency_ms = (time.time() - send_time) * 1000
                    err = await response.text()
                    return TTSResult(
                        success=False,
                        audio_data=None,
                        latency_ms=latency_ms,
                        file_size_bytes=0,
                        error_message=f"API Error {response.status}: {err}",
                        metadata={"provider": self.provider_id},
                    )
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id},
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id},
            )

    def get_available_voices(self) -> list:
        return self.config.supported_voices


class Falcon2TTSProvider(TTSProvider):
    """Murf Falcon speech/stream API with model=FALCON (dev or prod endpoint)."""

    def __init__(self, provider_id: str):
        if provider_id not in ("falcon_dev", "falcon_prod"):
            raise ValueError(f"Not a Falcon provider: {provider_id}")
        super().__init__(provider_id)

    def validate_request(self, request: TTSRequest) -> Tuple[bool, str]:
        if len(request.text) > self.config.max_chars:
            return False, f"Text exceeds maximum length of {self.config.max_chars} characters"
        if request.voice not in self.config.supported_voices:
            return False, (
                f"Voice '{request.voice}' not supported. "
                f"Available: {self.config.supported_voices}"
            )
        return True, ""

    @staticmethod
    def _locale_for_voice(voice_id: str) -> str:
        parts = voice_id.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return "en-US"

    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        start_time = time.time()
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={},
            )

        url = get_falcon_api_url(self.provider_id)
        headers = {**falcon_auth_headers(self.api_key), "Content-Type": "application/json"}
        payload = {
            "text": request.text,
            "voiceId": request.voice,
            "model": "FALCON",
            "locale": self._locale_for_voice(request.voice),
            "format": "WAV",
            "sampleRate": ARENA_SAMPLE_RATE,
            "channelType": "MONO",
        }

        try:
            timeout_s = falcon_synthesis_timeout(self.provider_id)
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                ) as response:
                    body, latency_ms = await self.read_body_ttfb(response, send_time)
                    if response.status == 200 and body:
                        fmt = "wav" if body[:4] == b"RIFF" else "mp3"
                        meta = {"provider": self.provider_id, "model": "FALCON", "voice": request.voice}
                        if fmt == "wav":
                            meta.update(_wav_meta())
                        else:
                            meta["format"] = fmt
                        return TTSResult(
                            success=True,
                            audio_data=body,
                            latency_ms=latency_ms,
                            file_size_bytes=len(body),
                            error_message=None,
                            metadata=meta,
                        )
                    env_hint = (
                        "FALCON_DEV_API_KEY" if self.provider_id == "falcon_dev"
                        else "FALCON_PROD_API_KEY"
                    )
                    return TTSResult(
                        success=False,
                        audio_data=None,
                        latency_ms=latency_ms,
                        file_size_bytes=0,
                        error_message=(
                            f"API Error {response.status}: {body[:500]!r}. "
                            f"Check {env_hint} and the endpoint URL for {self.provider_id}."
                        ),
                        metadata={"provider": self.provider_id},
                    )
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id},
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id},
            )

    def get_available_voices(self) -> list:
        return self.config.supported_voices


class OmniTTSProvider(Falcon2TTSProvider):
    """Backward-compatible alias for falcon_dev."""

    def __init__(self):
        super().__init__("falcon_dev")

class MurfZeroshotTTSProvider(TTSProvider):
    """Murf Zeroshot TTS provider implementation"""
    
    def __init__(self):
        super().__init__("murf_zeroshot")
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Murf Zeroshot API"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        # Dev endpoint uses api-key header (same as production)
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Extract locale from voice ID (e.g., "pa-IN-harman" -> "pa-IN", "en-US-alina" -> "en-US")
        # Voice IDs are in format: "{locale}-{voice-name}"
        voice_parts = request.voice.split("-")
        if len(voice_parts) >= 2:
            # Extract locale (first two parts: language and country)
            locale = f"{voice_parts[0]}-{voice_parts[1]}"
        else:
            # Fallback to en-US if we can't parse the locale
            locale = "en-US"
        
        # Murf Zeroshot API payload structure
        payload = {
            "text": request.text,
            "voiceId": request.voice,
            "multiNativeLocale": locale,
            "model": "FALCON",
            "format": "WAV",
            "sampleRate": 24000
        }
        
        # Add speed/rate if specified
        if request.speed and request.speed != 1.0:
            payload["rate"] = request.speed
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        audio_data = await response.read()
                        file_size = len(audio_data)
                        
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=file_size,
                            error_message=None,
                            metadata={
                                "provider": self.provider_id,
                                "model": "FALCON",
                                "voice": request.voice,
                                "format": request.format or "WAV"
                            }
                        )
                    else:
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text}",
                            metadata={"provider": self.provider_id}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available Murf Zeroshot voices"""
        return self.config.supported_voices

class DeepgramTTSProvider(TTSProvider):
    """Deepgram Aura 1 TTS provider implementation"""
    
    def __init__(self):
        super().__init__("deepgram")
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Deepgram TTS API"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Deepgram TTS API payload structure
        payload = {
            "text": request.text
        }
        
        # Add query parameters to URL
        params = {
            "model": request.voice,
            "encoding": "mp3" if request.format == "mp3" else "linear16"
        }
        
        # Only add sample_rate for non-MP3 formats
        if request.format != "mp3":
            params["sample_rate"] = "24000"
        
        # Build URL with parameters
        url_with_params = f"{self.config.base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                async with session.post(
                    url_with_params,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    if response.status == 200:
                        # Deepgram returns audio data directly
                        audio_data = await response.read()
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata={
                                "voice": request.voice,
                                "speed": request.speed,
                                "format": request.format,
                                "provider": self.provider_id,
                                "model": request.voice,
                                "sample_rate": 24000
                            }
                        )
                    else:
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text}",
                            metadata={"provider": self.provider_id}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available Deepgram voices"""
        return self.config.supported_voices

class DeepgramAura2TTSProvider(TTSProvider):
    """Deepgram Aura 2 TTS provider implementation"""
    
    def __init__(self):
        super().__init__("deepgram_aura2")
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Deepgram Aura 2 TTS API"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Deepgram Aura 2 TTS API payload structure
        payload = {
            "text": request.text
        }
        
        # For Deepgram Aura 2, the API expects the model parameter to be the full voice ID
        # Voice IDs are like "aura-2-asteria-en" - use this directly as the model parameter
        # According to Deepgram API docs, Aura 2 uses the full voice ID as the model parameter
        # Add query parameters to URL
        params = {
            "model": request.voice,
            "encoding": "linear16",
            "sample_rate": str(ARENA_SAMPLE_RATE),
        }
        
        # Build URL with parameters - use urllib.parse.urlencode for proper encoding
        import urllib.parse
        encoded_params = urllib.parse.urlencode(params)
        url_with_params = f"{self.config.base_url}?{encoded_params}"
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    url_with_params,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        pcm, latency_ms = await self.read_body_ttfb(response, send_time)
                        audio_data = pcm16_to_wav(pcm)
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata=_wav_meta({
                                "voice": request.voice,
                                "speed": request.speed,
                                "provider": self.provider_id,
                                "model": request.voice,
                            }),
                        )
                    else:
                        latency_ms = (time.time() - send_time) * 1000
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text}",
                            metadata={"provider": self.provider_id}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available Deepgram Aura 2 voices"""
        return self.config.supported_voices

class ElevenLabsFlashTTSProvider(TTSProvider):
    """ElevenLabs Flash TTS provider implementation"""
    
    def __init__(self):
        super().__init__("elevenlabs_flash")
        # Map friendly voice names to voice IDs (based on Artificial Analysis methodology)
        # Only voices listed on https://artificialanalysis.ai/text-to-speech/methodology
        # Turbo v2.5: Laura, Jessica, Liam, Elizabeth, Jarnathan, Dan, Nathaniel
        # Fallback voice IDs (from Artificial Analysis - may not work for all accounts)
        self.fallback_voice_id_map = {
            "Laura": "FGY2WhTYpPnrIDTdsKH5",
            "Jessica": "cgSgspJ2msm6clMCkdW9",
            "Liam": "TX3LPaxmHKxFdv7VOQHJ",
            "Elizabeth": "MF3mGyEYCl7XYWbV9V6O",
            "Jarnathan": "c6SfcYrb2t09NHXiT80T",
            "Dan": "TxGEqnHWrfWFTfGW9XjX",
            "Nathaniel": "N2lVS1w4EtoT3dr4eOWO"
        }
        # Will be populated with actual voice IDs from user's account
        self.voice_id_map = {}
        self._voices_fetched = False
    
    async def _fetch_voices_from_api(self):
        """Fetch available voices from ElevenLabs API and map them by name"""
        if self._voices_fetched:
            return
        
        try:
            headers = {"xi-api-key": self.api_key}
            voices_url = "https://api.elevenlabs.io/v1/voices"
            
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                async with session.get(
                    voices_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        voices_data = await response.json()
                        # Map voice names to their IDs
                        for voice in voices_data.get("voices", []):
                            voice_name = voice.get("name", "")
                            voice_id = voice.get("voice_id", "")
                            if voice_name and voice_id:
                                self.voice_id_map[voice_name] = voice_id
                        
                        # If we found voices, use them; otherwise fall back to hardcoded IDs
                        if not self.voice_id_map:
                            print("Warning: No voices found from API, using fallback IDs")
                            self.voice_id_map = self.fallback_voice_id_map.copy()
                        else:
                            print(f"Successfully fetched {len(self.voice_id_map)} voices from ElevenLabs API")
                            # Also check if any expected voices are missing
                            missing_voices = set(self.fallback_voice_id_map.keys()) - set(self.voice_id_map.keys())
                            if missing_voices:
                                print(f"Warning: Some expected voices not found in account: {missing_voices}")
                    else:
                        print(f"Failed to fetch voices from API (status {response.status}), using fallback IDs")
                        self.voice_id_map = self.fallback_voice_id_map.copy()
        except Exception as e:
            print(f"Error fetching voices from API: {e}, using fallback IDs")
            self.voice_id_map = self.fallback_voice_id_map.copy()
        
        self._voices_fetched = True
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using ElevenLabs TTS API"""
        start_time = time.time()
        
        # Fetch voices from API if not already fetched
        if not self._voices_fetched:
            await self._fetch_voices_from_api()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        # Validate voice exists in supported voices
        if request.voice not in self.config.supported_voices:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=f"Voice '{request.voice}' not in supported voices: {self.config.supported_voices}",
                metadata={"provider": self.provider_id}
            )
        
        # Get voice ID from map - try fetched IDs first, then fallback
        voice_id = self.voice_id_map.get(request.voice)
        if not voice_id:
            # Try fallback map
            voice_id = self.fallback_voice_id_map.get(request.voice)
            if not voice_id:
                return TTSResult(
                    success=False,
                    audio_data=None,
                    latency_ms=0,
                    file_size_bytes=0,
                    error_message=f"Voice '{request.voice}' not found in your ElevenLabs account. Available voices: {list(self.voice_id_map.keys())}",
                    metadata={"provider": self.provider_id, "available_voices": list(self.voice_id_map.keys())}
                )
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # ElevenLabs API payload structure
        payload = {
            "text": request.text,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        # Build URL with voice ID
        url = f"{self.config.base_url}/{voice_id}"
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    if response.status == 200:
                        # ElevenLabs returns audio data directly
                        audio_data = await response.read()
                        if len(audio_data) == 0:
                            return TTSResult(
                                success=False,
                                audio_data=None,
                                latency_ms=latency_ms,
                                file_size_bytes=0,
                                error_message="Empty audio response from API",
                                metadata={"provider": self.provider_id, "voice": request.voice, "voice_id": voice_id}
                            )
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata={
                                "voice": request.voice,
                                "voice_id": voice_id,
                                "model": "eleven_flash_v2_5",
                                "provider": self.provider_id,
                                "format": "mp3_44100_128"
                            }
                        )
                    elif response.status == 404:
                        # Voice ID not found - try to fetch available voices and find matching voice
                        try:
                            async with session.get(
                                f"{self.config.base_url.replace('/text-to-speech', '/voices')}",
                                headers={"xi-api-key": self.api_key},
                                timeout=aiohttp.ClientTimeout(total=10)
                            ) as voices_response:
                                if voices_response.status == 200:
                                    voices_data = await voices_response.json()
                                    # Try to find voice by name
                                    for voice in voices_data.get("voices", []):
                                        if voice.get("name") == request.voice:
                                            # Found matching voice, retry with correct ID
                                            correct_voice_id = voice.get("voice_id")
                                            if correct_voice_id:
                                                # Retry with correct voice ID
                                                retry_url = f"{self.config.base_url}/{correct_voice_id}"
                                                async with session.post(
                                                    retry_url,
                                                    headers=headers,
                                                    json=payload,
                                                    timeout=aiohttp.ClientTimeout(total=30)
                                                ) as retry_response:
                                                    if retry_response.status == 200:
                                                        audio_data = await retry_response.read()
                                                        if len(audio_data) > 0:
                                                            return TTSResult(
                                                                success=True,
                                                                audio_data=audio_data,
                                                                latency_ms=(time.time() - start_time) * 1000,
                                                                file_size_bytes=len(audio_data),
                                                                error_message=None,
                                                                metadata={
                                                                    "voice": request.voice,
                                                                    "voice_id": correct_voice_id,
                                                                    "model": "eleven_flash_v2_5",
                                                                    "provider": self.provider_id,
                                                                    "format": "mp3_44100_128"
                                                                }
                                                            )
                        except:
                            pass  # Fall through to error
                        
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"Voice '{request.voice}' not found in your account. Voice ID {voice_id} doesn't exist. Please check your ElevenLabs account has this voice available.",
                            metadata={"provider": self.provider_id, "voice": request.voice, "voice_id": voice_id, "status": response.status}
                        )
                    else:
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text[:200]} (Voice: {request.voice}, Voice ID: {voice_id})",
                            metadata={"provider": self.provider_id, "voice": request.voice, "voice_id": voice_id, "status": response.status}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Request timeout (Voice: {request.voice})",
                metadata={"provider": self.provider_id, "voice": request.voice}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)} (Voice: {request.voice}, Voice ID: {voice_id})",
                metadata={"provider": self.provider_id, "voice": request.voice, "voice_id": voice_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available ElevenLabs Flash voices - returns actual voices from account if fetched, otherwise config voices"""
        # If voices have been fetched from API, return those (account-specific voices)
        if self.voice_id_map:
            return list(self.voice_id_map.keys())
        # Otherwise return config voices (fallback)
        return self.config.supported_voices

class ElevenLabsV3TTSProvider(TTSProvider):
    """ElevenLabs v3 TTS provider implementation"""
    
    def __init__(self):
        super().__init__("elevenlabs_v3")
        # Map friendly voice names to voice IDs (based on Artificial Analysis methodology)
        # Only voices listed on https://artificialanalysis.ai/text-to-speech/methodology
        # Turbo v2.5: Laura, Jessica, Liam, Elizabeth, Jarnathan, Dan, Nathaniel
        # Fallback voice IDs (from Artificial Analysis - may not work for all accounts)
        self.fallback_voice_id_map = {
            "Laura": "FGY2WhTYpPnrIDTdsKH5",
            "Jessica": "cgSgspJ2msm6clMCkdW9",
            "Liam": "TX3LPaxmHKxFdv7VOQHJ",
            "Elizabeth": "MF3mGyEYCl7XYWbV9V6O",
            "Jarnathan": "c6SfcYrb2t09NHXiT80T",
            "Dan": "TxGEqnHWrfWFTfGW9XjX",
            "Nathaniel": "N2lVS1w4EtoT3dr4eOWO"
        }
        # Will be populated with actual voice IDs from user's account
        self.voice_id_map = {}
        self._voices_fetched = False
    
    async def _fetch_voices_from_api(self):
        """Fetch available voices from ElevenLabs API and map them by name"""
        if self._voices_fetched:
            return
        
        try:
            headers = {"xi-api-key": self.api_key}
            voices_url = "https://api.elevenlabs.io/v1/voices"
            
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                async with session.get(
                    voices_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        voices_data = await response.json()
                        # Map voice names to their IDs
                        for voice in voices_data.get("voices", []):
                            voice_name = voice.get("name", "")
                            voice_id = voice.get("voice_id", "")
                            if voice_name and voice_id:
                                self.voice_id_map[voice_name] = voice_id
                        
                        # If we found voices, use them; otherwise fall back to hardcoded IDs
                        if not self.voice_id_map:
                            print("Warning: No voices found from API, using fallback IDs")
                            self.voice_id_map = self.fallback_voice_id_map.copy()
                        else:
                            print(f"Successfully fetched {len(self.voice_id_map)} voices from ElevenLabs API")
                            # Also check if any expected voices are missing
                            missing_voices = set(self.fallback_voice_id_map.keys()) - set(self.voice_id_map.keys())
                            if missing_voices:
                                print(f"Warning: Some expected voices not found in account: {missing_voices}")
                    else:
                        print(f"Failed to fetch voices from API (status {response.status}), using fallback IDs")
                        self.voice_id_map = self.fallback_voice_id_map.copy()
        except Exception as e:
            print(f"Error fetching voices from API: {e}, using fallback IDs")
            self.voice_id_map = self.fallback_voice_id_map.copy()
        
        self._voices_fetched = True
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using ElevenLabs v3 TTS API"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )

        # Arena registry stores ElevenLabs voice_id strings directly.
        if request.voice in self.config.supported_voices:
            voice_id = request.voice
        else:
            if not self._voices_fetched:
                await self._fetch_voices_from_api()
            voice_id = self.voice_id_map.get(request.voice)
            if not voice_id:
                voice_id = self.fallback_voice_id_map.get(request.voice)
            if not voice_id:
                return TTSResult(
                    success=False,
                    audio_data=None,
                    latency_ms=0,
                    file_size_bytes=0,
                    error_message=f"Voice '{request.voice}' not found in your ElevenLabs account. Available voices: {list(self.voice_id_map.keys())}",
                    metadata={"provider": self.provider_id, "available_voices": list(self.voice_id_map.keys())}
                )
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # ElevenLabs Flash 2.5 API payload structure
        payload = {
            "text": request.text,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        # Build URL with voice ID — request raw PCM @ 24 kHz, wrap as WAV below.
        url = f"{self.config.base_url}/{voice_id}?output_format=pcm_{ARENA_SAMPLE_RATE}"
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        pcm, latency_ms = await self.read_body_ttfb(response, send_time)
                        audio_data = pcm16_to_wav(pcm)
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata=_wav_meta({
                                "voice": request.voice,
                                "voice_id": voice_id,
                                "model": "eleven_flash_v2_5",
                                "provider": self.provider_id,
                            }),
                        )
                    else:
                        latency_ms = (time.time() - send_time) * 1000
                        error_text = await response.text()
                        friendly = humanize_provider_error(
                            self.provider_id, response.status, error_text)
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=friendly,
                            metadata={
                                "provider": self.provider_id,
                                "http_status": response.status,
                                "cloud_blocked": "detected_unusual_activity" in error_text,
                            },
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available ElevenLabs v3 voices - returns actual voices from account if fetched, otherwise config voices"""
        # If voices have been fetched from API, return those (account-specific voices)
        if self.voice_id_map:
            return list(self.voice_id_map.keys())
        # Otherwise return config voices (fallback)
        return self.config.supported_voices

class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS provider implementation"""
    
    def __init__(self):
        super().__init__("openai")
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using OpenAI TTS API"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # OpenAI TTS API payload structure
        payload = {
            "model": "gpt-4o-mini-tts",  # GPT-4o Mini TTS model
            "input": request.text,
            "voice": request.voice.lower(),  # alloy, echo, fable, onyx, nova, shimmer
            "response_format": "wav",
            "speed": request.speed if request.speed else 1.0
        }
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        # OpenAI returns audio data directly
                        audio_data, latency_ms = await self.read_body_ttfb(response, send_time)
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata=_wav_meta({
                                "voice": request.voice,
                                "model": "gpt-4o-mini-tts",
                                "provider": self.provider_id,
                                "speed": payload["speed"],
                            }),
                        )
                    else:
                        latency_ms = (time.time() - send_time) * 1000
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text}",
                            metadata={"provider": self.provider_id}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available OpenAI voices"""
        return self.config.supported_voices

class CartesiaTTSProvider(TTSProvider):
    """Base class for Cartesia TTS providers"""
    
    def __init__(self, provider_id: str, model_id: str):
        super().__init__(provider_id)
        self.model_id = model_id
        # Map friendly voice names to Cartesia voice IDs
        self.voice_id_map = {
            "British Lady": "79a125e8-cd45-4c13-8a67-188112f4dd22",
            "Conversational Lady": "a0e99841-438c-4a64-b679-ae501e7d6091",
            "Classy British Man": "63ff761f-c1e8-414b-b969-d1833d1c870c",
            "Friendly Reading Man": "5619d38c-cf51-4d8e-9575-48f61a280413",
            "Midwestern Woman": "a3520a8f-226a-428d-9fcd-b0a4711a6829",
            "Professional Man": "41534e16-2966-4c6b-9670-111411def906",
            "Newsman": "daf747c6-6bc2-45c9-b3e6-d99d48c6697e"
        }
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Cartesia TTS API"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        # Arena registry stores Cartesia voice UUIDs directly.
        if request.voice in self.config.supported_voices:
            voice_id = request.voice
        else:
            voice_id = self.voice_id_map.get(request.voice)
            if not voice_id:
                return TTSResult(
                    success=False,
                    audio_data=None,
                    latency_ms=0,
                    file_size_bytes=0,
                    error_message=f"Voice '{request.voice}' not supported for Cartesia.",
                    metadata={"provider": self.provider_id},
                )
        
        headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json"
        }
        
        # Cartesia API payload structure
        payload = {
            "model_id": self.model_id,
            "transcript": request.text,
            "voice": {
                "mode": "id",
                "id": voice_id
            },
            "language": "en",
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": ARENA_SAMPLE_RATE,
            }
        }
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        # Cartesia returns audio data directly
                        audio_data, latency_ms = await self.read_body_ttfb(response, send_time)
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata=_wav_meta({
                                "voice": request.voice,
                                "voice_id": voice_id,
                                "model": self.model_id,
                                "provider": self.provider_id,
                            }),
                        )
                    else:
                        latency_ms = (time.time() - send_time) * 1000
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text}",
                            metadata={"provider": self.provider_id}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available Cartesia voices"""
        return self.config.supported_voices

class CartesiaSonic2Provider(CartesiaTTSProvider):
    """Cartesia Sonic 2.0 TTS provider"""
    
    def __init__(self):
        super().__init__("cartesia_sonic2", "sonic-2")

class CartesiaTurboProvider(CartesiaTTSProvider):
    """Cartesia Sonic Turbo TTS provider"""
    
    def __init__(self):
        super().__init__("cartesia_turbo", "sonic-turbo")

class CartesiaSonic3Provider(CartesiaTTSProvider):
    """Cartesia Sonic 3.5 TTS provider"""

    def __init__(self):
        super().__init__("cartesia_sonic3", "sonic-3.5")

class SarvamTTSProvider(TTSProvider):
    """Sarvam AI TTS provider implementation"""
    
    def __init__(self):
        super().__init__("sarvam")
        # Map voice IDs to Sarvam speaker names
        # Sarvam has: Male voices (abhilash, karun, hitesh) and Female voices (anushka, manisha, vidya, arya)
        self.voice_to_speaker_map = {
            "en-IN-male": "abhilash",      # Male English voice
            "en-IN-female": "anushka",     # Female English voice
            "hi-IN-male": "karun",         # Male Hindi voice
            "hi-IN-female": "manisha"      # Female Hindi voice
        }
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Sarvam AI API"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Determine language based on voice selection
        language = "en-IN" if "en-IN" in request.voice else "hi-IN"
        
        # Get speaker name from voice mapping - CRITICAL for gender selection
        speaker = self.voice_to_speaker_map.get(request.voice, "anushka")  # Default to female if not found
        
        # Log for debugging gender selection
        print(f"[SARVAM DEBUG] Voice: {request.voice} -> Speaker: {speaker}, Language: {language}")
        
        # Sarvam AI API payload structure - MUST include speaker parameter for correct voice selection
        payload = {
            "text": request.text,
            "model": "bulbul:v2",
            "language": language,
            "speaker": speaker  # This parameter is REQUIRED to select the correct voice/gender
        }
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    if response.status == 200:
                        # Check content type to determine response format
                        content_type = response.headers.get('content-type', '').lower()
                        
                        if 'application/json' in content_type:
                            # JSON response - might contain audio URL or base64 data
                            response_data = await response.json()
                            
                            if "audios" in response_data:
                                # Sarvam AI returns base64 encoded audio in 'audios' array
                                import base64
                                # audios is typically an array, get the first one
                                audio_base64 = response_data["audios"][0] if isinstance(response_data["audios"], list) else response_data["audios"]
                                audio_data = base64.b64decode(audio_base64)
                                return TTSResult(
                                    success=True,
                                    audio_data=audio_data,
                                    latency_ms=latency_ms,
                                    file_size_bytes=len(audio_data),
                                    error_message=None,
                                    metadata={
                                        "voice": request.voice,
                                        "language": language,
                                        "model": "bulbul:v2",
                                        "provider": self.provider_id,
                                        "format": "mp3",
                                        "request_id": response_data.get("request_id", "")
                                    }
                                )
                            elif "audioContent" in response_data:
                                # Base64 encoded audio data
                                import base64
                                audio_data = base64.b64decode(response_data["audioContent"])
                                return TTSResult(
                                    success=True,
                                    audio_data=audio_data,
                                    latency_ms=latency_ms,
                                    file_size_bytes=len(audio_data),
                                    error_message=None,
                                    metadata={
                                        "voice": request.voice,
                                        "language": language,
                                        "model": "bulbul:v2",
                                        "provider": self.provider_id,
                                        "format": "mp3"
                                    }
                                )
                            elif "audio" in response_data:
                                # Alternative base64 field name
                                import base64
                                audio_data = base64.b64decode(response_data["audio"])
                                return TTSResult(
                                    success=True,
                                    audio_data=audio_data,
                                    latency_ms=latency_ms,
                                    file_size_bytes=len(audio_data),
                                    error_message=None,
                                    metadata={
                                        "voice": request.voice,
                                        "language": language,
                                        "model": "bulbul:v2",
                                        "provider": self.provider_id,
                                        "format": "mp3"
                                    }
                                )
                            else:
                                return TTSResult(
                                    success=False,
                                    audio_data=None,
                                    latency_ms=latency_ms,
                                    file_size_bytes=0,
                                    error_message=f"Unexpected JSON response format: {list(response_data.keys())}",
                                    metadata={"provider": self.provider_id, "response": response_data}
                                )
                        else:
                            # Direct audio data response
                            audio_data = await response.read()
                            return TTSResult(
                                success=True,
                                audio_data=audio_data,
                                latency_ms=latency_ms,
                                file_size_bytes=len(audio_data),
                                error_message=None,
                                metadata={
                                    "voice": request.voice,
                                    "language": language,
                                    "model": "bulbul:v2",
                                    "provider": self.provider_id,
                                    "format": "mp3",
                                    "content_type": content_type
                                }
                            )
                    else:
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text}",
                            metadata={"provider": self.provider_id}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available Sarvam AI voices"""
        return self.config.supported_voices

class SarvamBulbulV3TTSProvider(TTSProvider):
    """Sarvam AI Bulbul v3 TTS provider implementation"""
    
    def __init__(self):
        super().__init__("sarvam_bulbul_v3")
        self.voice_to_speaker_map = {
            "en-IN-male": "aditya",
            "en-IN-male-2": "ashutosh",
            "en-IN-female": "priya",
            "en-IN-female-2": "neha",
            "hi-IN-male": "amit",
            "hi-IN-male-2": "rahul",
            "hi-IN-female": "ritu",
            "hi-IN-female-2": "pooja",
            "bn-IN-male": "kabir",
            "bn-IN-male-2": "rohan",
            "bn-IN-female": "shreya",
            "bn-IN-female-2": "ishita",
            "ta-IN-male": "tarun",
            "ta-IN-male-2": "varun",
            "ta-IN-female": "anand",
            "ta-IN-female-2": "tanya",
            "mr-IN-male": "advait",
            "mr-IN-male-2": "manan",
            "mr-IN-female": "roopa",
            "mr-IN-female-2": "rupali",
            "ml-IN-male": "gokul",
            "ml-IN-male-2": "vijay",
            "ml-IN-female": "shruti",
            "ml-IN-female-2": "suhani",
        }
        self._sarvam_languages = {
            "en-IN", "hi-IN", "bn-IN", "ta-IN", "mr-IN", "ml-IN",
        }
    
    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Sarvam AI Bulbul v3 API"""
        start_time = time.time()
        
        # Validate request
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=0,
                file_size_bytes=0,
                error_message=error_msg,
                metadata={}
            )
        
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        bits = request.voice.split("-")
        language = f"{bits[0]}-{bits[1]}" if len(bits) >= 2 else "en-IN"
        if language not in self._sarvam_languages:
            language = "en-IN"
        
        speaker = self.voice_to_speaker_map.get(request.voice, "priya")
        
        print(f"[SARVAM BULBUL V3 DEBUG] Voice: {request.voice} -> Speaker: {speaker}, Language: {language}")
        
        # Sarvam AI API payload structure - MUST include speaker parameter for correct voice selection
        payload = {
            "text": request.text,
            "model": "bulbul:v3",
            "language": language,
            "speaker": speaker,
            "speech_sample_rate": ARENA_SAMPLE_RATE,
            "output_audio_codec": "wav",
        }
        
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        # Check content type to determine response format
                        content_type = response.headers.get('content-type', '').lower()
                        body, latency_ms = await self.read_body_ttfb(response, send_time)
                        
                        if 'application/json' in content_type:
                            # JSON response - might contain audio URL or base64 data
                            import json as _json
                            response_data = _json.loads(body.decode('utf-8'))
                            
                            if "audios" in response_data:
                                # Sarvam AI returns base64 encoded audio in 'audios' array
                                import base64
                                # audios is typically an array, get the first one
                                audio_base64 = response_data["audios"][0] if isinstance(response_data["audios"], list) else response_data["audios"]
                                audio_data = base64.b64decode(audio_base64)
                                return TTSResult(
                                    success=True,
                                    audio_data=audio_data,
                                    latency_ms=latency_ms,
                                    file_size_bytes=len(audio_data),
                                    error_message=None,
                                    metadata=_wav_meta({
                                        "voice": request.voice,
                                        "language": language,
                                        "model": "bulbul:v3",
                                        "provider": self.provider_id,
                                        "request_id": response_data.get("request_id", ""),
                                    }),
                                )
                            elif "audioContent" in response_data:
                                # Base64 encoded audio data
                                import base64
                                audio_data = base64.b64decode(response_data["audioContent"])
                                return TTSResult(
                                    success=True,
                                    audio_data=audio_data,
                                    latency_ms=latency_ms,
                                    file_size_bytes=len(audio_data),
                                    error_message=None,
                                    metadata=_wav_meta({
                                        "voice": request.voice,
                                        "language": language,
                                        "model": "bulbul:v3",
                                        "provider": self.provider_id,
                                    }),
                                )
                            elif "audio" in response_data:
                                # Alternative base64 field name
                                import base64
                                audio_data = base64.b64decode(response_data["audio"])
                                return TTSResult(
                                    success=True,
                                    audio_data=audio_data,
                                    latency_ms=latency_ms,
                                    file_size_bytes=len(audio_data),
                                    error_message=None,
                                    metadata=_wav_meta({
                                        "voice": request.voice,
                                        "language": language,
                                        "model": "bulbul:v3",
                                        "provider": self.provider_id,
                                    }),
                                )
                            else:
                                return TTSResult(
                                    success=False,
                                    audio_data=None,
                                    latency_ms=latency_ms,
                                    file_size_bytes=0,
                                    error_message=f"Unexpected JSON response format: {list(response_data.keys())}",
                                    metadata={"provider": self.provider_id, "response": response_data}
                                )
                        else:
                            # Direct audio data response
                            audio_data = body
                            return TTSResult(
                                success=True,
                                audio_data=audio_data,
                                latency_ms=latency_ms,
                                file_size_bytes=len(audio_data),
                                error_message=None,
                                metadata=_wav_meta({
                                    "voice": request.voice,
                                    "language": language,
                                    "model": "bulbul:v3",
                                    "provider": self.provider_id,
                                    "content_type": content_type,
                                }),
                            )
                    else:
                        latency_ms = (time.time() - send_time) * 1000
                        error_text = await response.text()
                        return TTSResult(
                            success=False,
                            audio_data=None,
                            latency_ms=latency_ms,
                            file_size_bytes=0,
                            error_message=f"API Error {response.status}: {error_text}",
                            metadata={"provider": self.provider_id}
                        )
        
        except asyncio.TimeoutError:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message="Request timeout",
                metadata={"provider": self.provider_id}
            )
        except Exception as e:
            return TTSResult(
                success=False,
                audio_data=None,
                latency_ms=(time.time() - start_time) * 1000,
                file_size_bytes=0,
                error_message=f"Error: {str(e)}",
                metadata={"provider": self.provider_id}
            )
    
    def get_available_voices(self) -> list:
        """Get available Sarvam AI Bulbul v3 voices"""
        return self.config.supported_voices

class GoogleTTSProvider(TTSProvider):
    """Google Cloud Text-to-Speech (REST, API-key auth).

    The voice id encodes the BCP-47 languageCode as its first two hyphen
    segments (e.g. ``en-US-Neural2-D`` -> languageCode ``en-US``).
    """

    def __init__(self):
        super().__init__("google_tts")

    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        start_time = time.time()
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(False, None, 0, 0, error_msg, {})

        parts = request.voice.split("-")
        language_code = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "en-US"
        url = f"{self.config.base_url}?key={self.api_key}"
        payload = {
            "input": {"text": request.text},
            "voice": {"languageCode": language_code, "name": request.voice},
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": ARENA_SAMPLE_RATE,
            },
        }
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        import json as _json
                        raw, latency_ms = await self.read_body_ttfb(response, send_time)
                        body = _json.loads(raw.decode("utf-8"))
                        pcm = base64.b64decode(body.get("audioContent", ""))
                        if not pcm:
                            return TTSResult(False, None, latency_ms, 0,
                                             "Empty audioContent from Google", {"provider": self.provider_id})
                        audio_data = pcm16_to_wav(pcm)
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata=_wav_meta({
                                "provider": self.provider_id,
                                "model": "neural2",
                                "voice": request.voice,
                                "language": language_code,
                            }),
                        )
                    latency_ms = (time.time() - send_time) * 1000
                    err = await response.text()
                    return TTSResult(False, None, latency_ms, 0,
                                     f"API Error {response.status}: {err}", {"provider": self.provider_id})
        except asyncio.TimeoutError:
            return TTSResult(False, None, (time.time() - start_time) * 1000, 0,
                             "Request timeout", {"provider": self.provider_id})
        except Exception as e:
            return TTSResult(False, None, (time.time() - start_time) * 1000, 0,
                             f"Error: {str(e)}", {"provider": self.provider_id})

    def get_available_voices(self) -> list:
        return self.config.supported_voices


class AzureTTSProvider(TTSProvider):
    """Microsoft Azure Cognitive Services TTS (SSML, region + key auth).

    Needs AZURE_SPEECH_KEY (the api key) and AZURE_SPEECH_REGION (e.g.
    ``eastus``). The voice id encodes the locale (``en-US-JennyNeural``).
    """

    def __init__(self):
        super().__init__("azure_tts")
        self.region = os.getenv("AZURE_SPEECH_REGION", "").strip()

    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        start_time = time.time()
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(False, None, 0, 0, error_msg, {})
        if not self.region:
            return TTSResult(False, None, 0, 0,
                             "AZURE_SPEECH_REGION not set", {"provider": self.provider_id})

        parts = request.voice.split("-")
        locale = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "en-US"
        url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
            "User-Agent": "voice-arena",
        }
        ssml = (
            f"<speak version='1.0' xml:lang='{locale}'>"
            f"<voice xml:lang='{locale}' name='{request.voice}'>"
            f"{_xml_escape(request.text)}</voice></speak>"
        )
        try:
            async with aiohttp.ClientSession(connector=get_connector()) as session:
                send_time = time.time()
                async with session.post(
                    url, headers=headers, data=ssml.encode("utf-8"),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        audio_data, latency_ms = await self.read_body_ttfb(response, send_time)
                        return TTSResult(
                            success=True,
                            audio_data=audio_data,
                            latency_ms=latency_ms,
                            file_size_bytes=len(audio_data),
                            error_message=None,
                            metadata=_wav_meta({
                                "provider": self.provider_id,
                                "model": "neural",
                                "voice": request.voice,
                                "language": locale,
                            }),
                        )
                    latency_ms = (time.time() - send_time) * 1000
                    err = await response.text()
                    return TTSResult(False, None, latency_ms, 0,
                                     f"API Error {response.status}: {err}", {"provider": self.provider_id})
        except asyncio.TimeoutError:
            return TTSResult(False, None, (time.time() - start_time) * 1000, 0,
                             "Request timeout", {"provider": self.provider_id})
        except Exception as e:
            return TTSResult(False, None, (time.time() - start_time) * 1000, 0,
                             f"Error: {str(e)}", {"provider": self.provider_id})

    def get_available_voices(self) -> list:
        return self.config.supported_voices


class AmazonPollyTTSProvider(TTSProvider):
    """Amazon Polly (neural engine) via boto3 (SigV4 from AWS_* env creds).

    Reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION from the
    environment (boto3's default credential chain). The voice id is a Polly
    VoiceId (``Matthew``, ``Joanna``, ...) which implies the language.
    """

    def __init__(self):
        super().__init__("amazon_polly")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1").strip() or "us-east-1"

    async def generate_speech(self, request: TTSRequest) -> TTSResult:
        start_time = time.time()
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return TTSResult(False, None, 0, 0, error_msg, {})

        def _synthesize() -> bytes:
            import boto3
            client = boto3.client("polly", region_name=self.aws_region)
            resp = client.synthesize_speech(
                Text=request.text,
                OutputFormat="pcm",
                SampleRate=str(ARENA_SAMPLE_RATE),
                VoiceId=request.voice,
                Engine="neural",
            )
            return pcm16_to_wav(resp["AudioStream"].read())

        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(None, _synthesize)
            latency_ms = (time.time() - start_time) * 1000
            if not audio_data:
                return TTSResult(False, None, latency_ms, 0,
                                 "Empty audio from Polly", {"provider": self.provider_id})
            return TTSResult(
                success=True,
                audio_data=audio_data,
                latency_ms=latency_ms,
                file_size_bytes=len(audio_data),
                error_message=None,
                metadata=_wav_meta({
                    "provider": self.provider_id,
                    "model": "neural",
                    "voice": request.voice,
                }),
            )
        except Exception as e:
            return TTSResult(False, None, (time.time() - start_time) * 1000, 0,
                             f"Error: {str(e)}", {"provider": self.provider_id})

    def get_available_voices(self) -> list:
        return self.config.supported_voices


class TTSProviderFactory:
    """Factory for creating TTS providers.

    Generalized to cover every registered arena provider. The registry
    (config.TTS_PROVIDERS) is the source of truth for which providers exist;
    this map binds each provider_id to its implementation class.
    """

    # provider_id -> implementation class (or zero-arg factory callable).
    # Only "best model per vendor" providers are registered for the arena.
    # (Legacy classes for older models remain in this module for reference but
    # are intentionally not wired up here.)
    _PROVIDER_CLASSES = {
        "falcon_dev": lambda: Falcon2TTSProvider("falcon_dev"),
        "falcon_prod": lambda: Falcon2TTSProvider("falcon_prod"),
    }

    @staticmethod
    def create_provider(provider_id: str) -> TTSProvider:
        """Create a TTS provider instance for any registered provider."""
        factory = TTSProviderFactory._PROVIDER_CLASSES.get(provider_id)
        if factory is None:
            raise ValueError(f"Unknown provider: {provider_id}")
        return factory()

    @staticmethod
    def get_available_providers() -> list:
        """Provider IDs known to both the registry and this factory."""
        return [pid for pid in TTS_PROVIDERS.keys()
                if pid in TTSProviderFactory._PROVIDER_CLASSES]

    @staticmethod
    def create_all_providers() -> Dict[str, TTSProvider]:
        """Instantiate every registered provider that can be constructed.

        Providers missing API keys / endpoints raise during construction and
        are skipped (they simply won't be scheduled).
        """
        providers = {}
        for provider_id in TTSProviderFactory.get_available_providers():
            try:
                providers[provider_id] = TTSProviderFactory.create_provider(provider_id)
            except Exception as e:
                print(f"Skipping provider {provider_id}: {e}")
        return providers

    @staticmethod
    def create_configured_providers() -> Dict[str, TTSProvider]:
        """Instantiate only providers whose credentials are configured."""
        from config import is_provider_configured
        providers = {}
        for provider_id in TTSProviderFactory.get_available_providers():
            if not is_provider_configured(provider_id):
                continue
            try:
                providers[provider_id] = TTSProviderFactory.create_provider(provider_id)
            except Exception as e:
                print(f"Skipping provider {provider_id}: {e}")
        return providers
