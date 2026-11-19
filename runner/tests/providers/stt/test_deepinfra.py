# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coval_bench.providers.stt.deepinfra (DeepInfraSTTProvider)."""

from __future__ import annotations

import io
import wave
from collections.abc import Callable

import httpx2
import pytest

from coval_bench.config import Settings
from coval_bench.providers.base import TranscriptionResult
from coval_bench.providers.stt import deepinfra
from coval_bench.providers.stt.deepinfra import DeepInfraSTTProvider
from coval_bench.registries.provider_keys import PROVIDER_ENV
from tests.providers.stt.conftest import make_settings

_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    pcm: bytes,
    handle: Callable[[httpx2.Request], httpx2.Response],
) -> TranscriptionResult:
    async with httpx2.AsyncClient(
        base_url="https://api.deepinfra.com", transport=httpx2.MockTransport(handle)
    ) as client:
        monkeypatch.setattr(deepinfra, "get_shared_client", lambda *args: client)
        provider = DeepInfraSTTProvider(settings, model=_MODEL)
        return await provider.measure_ttft(pcm, 1, 2, 16000, 0.1)


@pytest.mark.asyncio
async def test_deepinfra_success_and_wire_shape(
    monkeypatch: pytest.MonkeyPatch, stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"text": "Hello world, how are you?"})

    result = await _run(monkeypatch, stt_settings, audio_pcm_bytes, handle)

    assert result.error is None
    assert result.provider == "deepinfra-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    assert result.complete_transcript == "Hello world, how are you?"
    assert result.word_count == 5
    assert result.ttft_seconds is None
    assert result.audio_to_final_seconds is None
    assert result.total_time is not None
    assert result.partial_transcripts == []

    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "https://api.deepinfra.com/v1/openai/audio/transcriptions"
    assert request.headers["Authorization"] == "Bearer test-api-key-00000000000000000000"
    body = request.read()
    for field, value in (("model", _MODEL), ("language", "en"), ("response_format", "json")):
        assert f'name="{field}"\r\n\r\n{value}'.encode() in body
    wav_start = body.index(b"RIFF")
    with wave.open(io.BytesIO(body[wav_start:]), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.readframes(wav.getnframes()) == audio_pcm_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_deepinfra_http_error_fails_the_clip(
    monkeypatch: pytest.MonkeyPatch, stt_settings: Settings, audio_pcm_bytes: bytes, status: int
) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"detail": "nope"})

    result = await _run(monkeypatch, stt_settings, audio_pcm_bytes, handle)

    assert result.error is not None and result.error.startswith(f"HTTP {status}")
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


@pytest.mark.asyncio
async def test_deepinfra_empty_transcript_scores_as_failed(
    monkeypatch: pytest.MonkeyPatch, stt_settings: Settings, audio_pcm_bytes: bytes
) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"text": ""})

    result = await _run(monkeypatch, stt_settings, audio_pcm_bytes, handle)

    assert result.error is None
    assert result.complete_transcript is None
    assert result.ttft_seconds is None


def test_deepinfra_construction_guards() -> None:
    with pytest.raises(ValueError, match="deepinfra_api_key is required"):
        DeepInfraSTTProvider(make_settings(deepinfra_api_key=None), model=_MODEL)
    with pytest.raises(ValueError, match="unsupported DeepInfra STT model"):
        DeepInfraSTTProvider(make_settings(), model="some/other-model")


def test_deepinfra_has_a_provider_env_entry() -> None:
    assert PROVIDER_ENV["deepinfra"] == "DEEPINFRA_API_KEY"
