// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_codec.h"

#include <algorithm>

namespace qwen3_tts {

namespace mx = mlx::core;

namespace {

// Note (Jiaxin Deng): the CUDA vocoder decodes a whole request in 300-frame
// chunks, each with up to 25 frames of left context.
constexpr int kChunkFrames = 300;
constexpr int kLeftContextFrames = 25;

} // namespace

SpeechStream::SpeechStream(const SpeechDecoder &decoder, bool streamed,
                           int first_chunk_frames)
    : decoder_(decoder), streamed_(streamed),
      first_chunk_frames_(
          first_chunk_frames > 0
              ? std::min(first_chunk_frames, kStreamedChunkFrames)
              : kStreamedChunkFrames),
      decoder_state_(streamed ? decoder.NewStreamState() : std::nullopt) {}

int SpeechStream::ChunkFrames() const {
  if (!streamed_) {
    return kChunkFrames;
  } else if (emitted_frames_ == 0) {
    return first_chunk_frames_;
  } else {
    return kStreamedChunkFrames;
  }
}

std::vector<float> SpeechStream::Push(const mx::array &frame_codes) {
  frames_.push_back(frame_codes);
  if (frame_codes.data<int32_t>()[0] > 0) {
    ++voiced_frames_;
  } else {
  }
  const int chunk_frames = ChunkFrames();
  if (static_cast<int>(frames_.size()) - emitted_frames_ < chunk_frames) {
    return {};
  } else if (streamed_) {
    return DecodeChunk(emitted_frames_ + chunk_frames);
  } else {
    const std::vector<float> chunk =
        DecodeChunk(emitted_frames_ + chunk_frames);
    held_samples_.insert(held_samples_.end(), chunk.begin(), chunk.end());
    return {};
  }
}

std::vector<float> SpeechStream::Finish() {
  std::vector<float> samples = std::move(held_samples_);
  while (emitted_frames_ < static_cast<int>(frames_.size())) {
    const std::vector<float> chunk = DecodeChunk(std::min(
        static_cast<int>(frames_.size()), emitted_frames_ + ChunkFrames()));
    samples.insert(samples.end(), chunk.begin(), chunk.end());
  }
  // Note (Jiaxin Deng): as the CUDA vocoder's whole-request decode, the audio
  // keeps one frame of samples per frame whose first code is not 0.
  const size_t kept_samples =
      static_cast<size_t>(voiced_frames_) * decoder_.samples_per_frame();
  if (!streamed_ && kept_samples < samples.size()) {
    samples.resize(kept_samples);
  } else {
  }
  return samples;
}

std::vector<float> SpeechStream::DecodeChunk(int end_frame) {
  if (decoder_state_) {
    return DecodeIncremental(end_frame);
  } else {
  }
  const int context_frames = std::min(kLeftContextFrames, emitted_frames_);
  const int start_frame = emitted_frames_ - context_frames;
  const std::vector<mx::array> window(frames_.begin() + start_frame,
                                      frames_.begin() + end_frame);
  const int samples_per_frame = decoder_.samples_per_frame();
  const mx::array audio =
      mx::slice(decoder_.Decode(mx::stack(window, 0)),
                {context_frames * samples_per_frame},
                {(end_frame - start_frame) * samples_per_frame});
  mx::eval(audio);
  emitted_frames_ = end_frame;
  std::vector<float> samples(audio.data<float>(),
                             audio.data<float>() + audio.size());
  // Note (Jiaxin Deng): a chunk's upsampling buffers run to gigabytes and
  // are not needed again until the next chunk, so MLX's cache returns them.
  mx::clear_cache();
  return samples;
}

std::vector<float> SpeechStream::DecodeIncremental(int end_frame) {
  std::vector<float> samples;
  const int samples_per_frame = decoder_.samples_per_frame();
  while (emitted_frames_ < end_frame) {
    const int start = decoder_state_->decoded_frames;
    const int offset = start % kStreamedChunkFrames;
    const int step_frames = (offset == 0 ? kDefaultFirstChunkFrames
                                         : kStreamedChunkFrames - offset);
    const int stop = std::min(end_frame, start + step_frames);
    const std::vector<mx::array> block(frames_.begin() + start,
                                       frames_.begin() + stop);
    mx::array codes = mx::stack(block, 0);
    std::optional<SpeechDecoderState> preview;
    if (stop - start < step_frames) {
      // Only emit real samples. Keep the calculation shape on partial and
      // final packets so a different first packet produces identical PCM.
      preview = *decoder_state_;
      codes = mx::concatenate(
          {codes, mx::broadcast_to(block.back(), {step_frames - (stop - start),
                                                  codes.shape(1)})},
          0);
    } else {
    }
    SpeechDecoderState &state = preview ? *preview : *decoder_state_;
    const mx::array audio =
        mx::slice(decoder_.Decode(codes, &state),
                  {(emitted_frames_ - start) * samples_per_frame},
                  {(stop - start) * samples_per_frame});
    std::vector<mx::array> outputs = {audio};
    if (!preview) {
      for (const auto &[prefix, buffer] : state.convolution_history) {
        outputs.push_back(buffer);
      }
      for (const auto &history : state.attention) {
        outputs.push_back(history->first);
        outputs.push_back(history->second);
      }
    } else {
    }
    mx::eval(outputs);
    samples.insert(samples.end(), audio.data<float>(),
                   audio.data<float>() + audio.size());
    emitted_frames_ = stop;
  }
  mx::clear_cache();
  return samples;
}

} // namespace qwen3_tts
