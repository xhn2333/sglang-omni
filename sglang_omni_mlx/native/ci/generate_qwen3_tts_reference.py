# SPDX-License-Identifier: Apache-2.0
"""Capture seeded CustomVoice prompt IDs, codec frames and WAVs with mlx-audio 0.5.8.

Use the serving contract's whole-text prompt, explicit Chinese language and full repetition history. Keep the wheel's model, sampler and speech decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from functools import partialmethod
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model
from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer
from mlx_audio.tts.utils import load
from provision import CI_DIRECTORY, sha256
from speech_golden import SpeechCase

REFERENCE_PREPARE = Model._prepare_generation_inputs
REFERENCE_DECODE = Qwen3TTSSpeechTokenizer.decode


def generate_case(model: Model, case: SpeechCase, directory: Path) -> None:
    config = model.config.talker_config
    speaker = case["speaker"].lower()
    language = case["language"].lower()
    dialect = config.spk_is_dialect[speaker]
    effective_language = dialect if language == "auto" and dialect else language
    if effective_language == "auto":
        prefix = [
            config.codec_nothink_id,
            config.codec_think_bos_id,
            config.codec_think_eos_id,
        ]
    else:
        prefix = [
            config.codec_think_id,
            config.codec_think_bos_id,
            config.codec_language_id[effective_language],
            config.codec_think_eos_id,
        ]
    prefix.extend([config.spk_id[speaker], config.codec_pad_id, config.codec_bos_id])
    text_ids = model.tokenizer.encode(
        f"<|im_start|>assistant\n{case['text']}<|im_end|>\n<|im_start|>assistant\n"
    )
    frame_codes: list[list[int]] = []

    def whole_text(
        instance: Model, text: str, language: str, speaker: str, instruct: str | None
    ) -> tuple[mx.array, mx.array, mx.array]:
        prefix_embeddings, trailing_text, text_pad = REFERENCE_PREPARE(
            instance, text=text, language=language, speaker=speaker, instruct=instruct
        )
        # Note (Codex): projecting a single token separately can select a different BF16 kernel.
        text_embeddings = instance.talker.text_projection(
            instance.talker.get_text_embeddings()(mx.array([text_ids]))
        )
        first_text = text_embeddings[:, 3:4]
        codec = instance.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )
        full_text = mx.concatenate([first_text, trailing_text], axis=1)
        embeddings = mx.concatenate(
            [
                prefix_embeddings[:, :-1],
                full_text + codec[:, :1],
                text_pad + codec[:, 1:],
            ],
            axis=1,
        )
        return embeddings, trailing_text[:, :0], text_pad

    def capture_codes(
        tokenizer: Qwen3TTSSpeechTokenizer, codes: mx.array
    ) -> tuple[mx.array, mx.array]:
        frame_codes.extend(codes.tolist()[0])
        return REFERENCE_DECODE(tokenizer, codes)

    dialects = (
        {**config.spk_is_dialect, speaker: False}
        if language == "chinese"
        else config.spk_is_dialect
    )
    mx.random.seed(case["seed"])
    with (
        patch.object(Model, "_prepare_generation_inputs", whole_text),
        patch.object(
            Model,
            "_sample_token",
            partialmethod(Model._sample_token, repetition_context_size=0),
        ),
        patch.object(Qwen3TTSSpeechTokenizer, "decode", capture_codes),
        patch.object(config, "spk_is_dialect", dialects),
    ):
        [result] = list(
            model.generate_custom_voice(
                text=case["text"],
                speaker=speaker,
                language=language,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.05,
                max_tokens=case["max_new_tokens"],
            )
        )
    samples = np.array(result.audio.astype(mx.float32))
    if not np.isfinite(samples).all():
        raise ValueError("Reference audio contains nonfinite samples")
    else:
        pass
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    directory.mkdir(parents=True, exist_ok=True)
    with wave.open(str(directory / "reference.wav"), "wb") as writer:
        writer.setparams((1, 2, model.sample_rate, 0, "NONE", ""))
        writer.writeframes(pcm)
    (directory / "reference.codes.json").write_text(
        json.dumps(
            {
                "prompt_ids": prefix + text_ids,
                "codes": frame_codes,
                "finish_reason": (
                    "length" if len(frame_codes) == case["max_new_tokens"] else "stop"
                ),
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            }
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        required=True,
        help="Read the model and requests from this golden file.",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    for package, expected in {
        "mlx-audio": "0.5.8",
        "mlx": "0.32.3",
        "mlx-metal": "0.32.3",
    }.items():
        if version(package) != expected:
            raise SystemExit(f"{package} {expected} is required")
        else:
            pass
    inputs = json.loads(arguments.golden.read_text())
    pin = json.loads((CI_DIRECTORY / "model_pins.json").read_text())[inputs["model"]]
    for filename, digest in pin["files"].items():
        if sha256(arguments.model_path / filename) != digest:
            raise SystemExit(f"Checkpoint hash mismatch: {filename}")
        else:
            pass
    model = load(str(arguments.model_path.resolve()))
    if model.tokenizer is None or model.speech_tokenizer is None:
        raise RuntimeError("mlx-audio did not load the text and speech tokenizers")
    else:
        pass
    for name, case in inputs["cases"].items():
        generate_case(model, case, arguments.output_directory / name)


if __name__ == "__main__":
    main()
