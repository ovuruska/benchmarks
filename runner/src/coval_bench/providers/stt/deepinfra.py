# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepInfra batch speech-to-text provider (OpenAI-compatible file transcription).

Wire protocol: HTTP, ``POST https://api.deepinfra.com/v1/openai/audio/transcriptions``
with multipart ``file``, ``model``, ``language`` and ``response_format=json``, and
``Authorization: Bearer <key>``.

This is a batch provider: the clip is uploaded whole and the transcript comes back
in one response, with no partials. TTFT and audio-to-final are not reported: the
upload round-trip is not comparable to the realtime-paced streaming metrics, so
clips are scored on the transcript (WER) only. ``total_time`` is still recorded.
"""

from __future__ import annotations

import io
import wave
from typing import ClassVar

from coval_bench.config import Settings
from coval_bench.providers._http_session import get_shared_client
from coval_bench.providers.base import STTProvider, TranscriptionResult
from coval_bench.providers.stt._transcript_utils import finalize_transcript

_BASE_URL = "https://api.deepinfra.com"
_PATH = "/v1/openai/audio/transcriptions"
_DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
# A whole clip is transcribed per request; long clips need more than the pool's 30 s default.
_TIMEOUT_S = 120.0


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


class DeepInfraSTTProvider(STTProvider):
    """DeepInfra file transcription over the OpenAI-compatible endpoint."""

    _VALID_MODELS: ClassVar[frozenset[str]] = frozenset({_DEFAULT_MODEL})

    def __init__(self, settings: Settings, model: str = _DEFAULT_MODEL) -> None:
        super().__init__(settings, model)
        if not self._model_supported(model):
            raise ValueError(f"unsupported DeepInfra STT model: {model}")
        self._api_key = self._require_secret("deepinfra_api_key")

    @property
    def name(self) -> str:
        return f"deepinfra-{self._model.rsplit('/', 1)[-1]}"

    async def _transcribe(
        self,
        audio_data: bytes,
        sample_rate: int,
        realtime_resolution: float,
        result: TranscriptionResult,
    ) -> None:
        client = get_shared_client("deepinfra", _BASE_URL)
        response = await client.post(
            _PATH,
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
            files={"file": ("audio.wav", _wav_bytes(audio_data, sample_rate), "audio/wav")},
            data={"model": self._model, "language": "en", "response_format": "json"},
            timeout=_TIMEOUT_S,
        )
        if response.status_code != 200:
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
            return
        text = str(response.json().get("text") or "")
        finalize_transcript(result, explicit_transcript=text, partial_fallback="none")
