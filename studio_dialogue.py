#!/usr/bin/env python3
"""
Studio-style dialogue cleanup for a video's audio track.

Pipeline (order matters):
  1. ffmpeg  : extract audio losslessly (24-bit WAV, mono)
  2. DSP pre : high-pass + hum notches (helps the AI model; tonal hum is its weak spot)
  3. AI      : DeepFilterNet (fast, natural) or Resemble Enhance (stronger denoise + dereverb)
  4. DSP post: de-mud / presence / air EQ, compression
  5. Loudness: integrated -16 LUFS (ITU-R BS.1770 via pyloudnorm)
  6. Limiter : ceiling ~ -1 dBTP LAST, so nothing downstream can push peaks back up
  7. ffmpeg  : remux with the original video stream copied untouched

Install:
    pip install pedalboard pyloudnorm numpy
    pip install resemble-enhance        # engine "resemble"  (GPU strongly recommended)
    pip install deepfilternet           # engine "deepfilter" (fine on CPU)
    ffmpeg must be on PATH

Usage:
    python studio_dialogue.py talk.mp4 --engine resemble --hum 50 --mux
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from pedalboard import (Compressor, HighpassFilter, HighShelfFilter, Limiter,
                        PeakFilter, Pedalboard)
from pedalboard.io import AudioFile


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract(video: Path, wav: Path, mono: bool) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vn", "-map", "0:a:0"]
    if mono:
        cmd += ["-ac", "1"]
    run(cmd + ["-ar", "44100", "-c:a", "pcm_s24le", str(wav)])


def read(path: Path) -> tuple[np.ndarray, int]:
    with AudioFile(str(path)) as f:
        return f.read(f.frames), f.samplerate          # shape: (channels, samples)


def write(path: Path, audio: np.ndarray, sr: int) -> None:
    with AudioFile(str(path), "w", sr, audio.shape[0], bit_depth=24) as f:
        f.write(audio)


def pre_filter(audio: np.ndarray, sr: int, hum_hz: int) -> np.ndarray:
    plugins = [HighpassFilter(80)]
    if hum_hz:  # narrow notches at fundamental + harmonics (50 Hz mains in Belgium, 60 in the US)
        for k in (1, 2, 3):
            plugins.append(PeakFilter(hum_hz * k, -20.0, 30.0))
    return Pedalboard(plugins)(audio, sr)


def ai_enhance(engine: str, src: Path, workdir: Path) -> Path:
    in_dir, out_dir = workdir / "in", workdir / "out"
    in_dir.mkdir(), out_dir.mkdir()
    shutil.copy(src, in_dir / "dialogue.wav")
    if engine == "resemble":
        run(["resemble_enhance", str(in_dir), str(out_dir)])   # add --denoise_only for a gentler result
    else:
        run(["deepFilter", str(in_dir / "dialogue.wav"), "-o", str(out_dir)])
    outs = sorted(out_dir.glob("*.wav"))
    if not outs:
        sys.exit("AI enhancer produced no output")
    return outs[0]


def post_chain(audio: np.ndarray, sr: int) -> np.ndarray:
    board = Pedalboard([
        HighpassFilter(70),
        PeakFilter(250, -2.5, 1.0),        # take out boxiness / proximity mud
        PeakFilter(3500, +2.5, 0.9),       # vocal presence / intelligibility
        HighShelfFilter(10000, +2.0, 0.7), # air (keep small: enhancers can already sound bright)
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=10, release_ms=120),
    ])
    return board(audio, sr)


def loudness_and_limit(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    data = audio.T.astype(np.float64)                      # pyloudnorm wants (samples, channels)
    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(data)
    print(f"  integrated loudness: {measured:.1f} LUFS -> {target_lufs:.1f} LUFS")
    data = pyln.normalize.loudness(data, measured, target_lufs)
    limited = Pedalboard([Limiter(threshold_db=-1.5, release_ms=100)])(data.T.astype(np.float32), sr)
    return limited


def mux(video: Path, wav: Path, out: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--engine", choices=["resemble", "deepfilter"], default="resemble")
    ap.add_argument("--hum", type=int, choices=[0, 50, 60], default=50, help="mains hum fundamental (0 = off)")
    ap.add_argument("--lufs", type=float, default=-16.0, help="target integrated loudness")
    ap.add_argument("--keep-stereo", action="store_true")
    ap.add_argument("--mux", action="store_true")
    args = ap.parse_args()

    out_wav = args.input.with_name(f"{args.input.stem}_studio.wav")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw.wav"

        print("1/5 extract");   extract(args.input, raw, mono=not args.keep_stereo)
        audio, sr = read(raw)

        print("2/5 pre-filter"); write(tmp / "pre.wav", pre_filter(audio, sr, args.hum), sr)

        print(f"3/5 AI enhance ({args.engine})")
        enhanced = ai_enhance(args.engine, tmp / "pre.wav", tmp)
        audio, sr = read(enhanced)                       # engine may change the sample rate

        print("4/5 EQ + compression"); audio = post_chain(audio, sr)
        print("5/5 loudness + limiter"); audio = loudness_and_limit(audio, sr, args.lufs)

        write(out_wav, audio, sr)
        print(f"Wrote {out_wav}")

    if args.mux:
        out_video = args.input.with_name(f"{args.input.stem}_studio{args.input.suffix}")
        mux(args.input, out_wav, out_video)
        print(f"Wrote {out_video}")


if __name__ == "__main__":
    main()
