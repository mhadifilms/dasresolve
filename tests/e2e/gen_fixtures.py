#!/usr/bin/env python3
"""Generate synthetic test footage for the DasGrain end-to-end Resolve test.

Produces three short ProRes 4444 movies (12-bit, alpha) so Resolve's
Fusion can ingest them through MediaIn after they're imported into the
media pool and dropped on a timeline.

Layout under ``tests/e2e/fixtures/``::

    plate.mov       grainy plate
    degrained.mov   clean version of the same plate
    comp.mov        clean variation (the "comp" we want to re-grain)
    preview/*.png   8-bit single-frame sanity previews

The frames carry a luminance ramp (so the response curve has dynamic range
to fit) plus a subtle moving feature (so frame-to-frame differences are
non-zero). The grain is per-channel Gaussian noise scaled by a luminance
curve so the analyser has something interesting to recover.

This script depends only on ``numpy``, ``imageio`` and a system ``ffmpeg``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURES = HERE / "fixtures"


def luma_ramp(h: int, w: int) -> np.ndarray:
    """Horizontal luminance ramp with a soft vertical falloff."""
    x = np.linspace(0.05, 0.95, w, dtype=np.float32)
    y = np.linspace(0.7, 1.0, h, dtype=np.float32)
    return (x[None, :] * y[:, None]).astype(np.float32)


def clean_frame(h: int, w: int, t: float) -> np.ndarray:
    """A clean, animated test frame in linear float."""
    base = luma_ramp(h, w)
    # A moving "blob" so frame-to-frame differences are non-zero.
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h, dtype=np.float32),
                         np.linspace(-1.0, 1.0, w, dtype=np.float32),
                         indexing="ij")
    cx = 0.4 * np.cos(2.0 * np.pi * t)
    cy = 0.4 * np.sin(2.0 * np.pi * t)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.05).astype(np.float32)
    base = base + 0.15 * blob
    base = np.clip(base, 0.0, 1.0)
    rgb = np.stack([base * 0.95, base * 1.00, base * 0.85], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def comp_recolour(frame: np.ndarray) -> np.ndarray:
    """A simple "regrade" of the clean plate so comp != degrained."""
    # Slightly cool and lift the shadows.
    out = frame.copy()
    out[..., 0] = np.clip(out[..., 0] * 0.97 + 0.02, 0.0, 1.0)  # red
    out[..., 1] = np.clip(out[..., 1] * 1.02, 0.0, 1.0)         # green
    out[..., 2] = np.clip(out[..., 2] * 1.05 + 0.01, 0.0, 1.0)  # blue
    return out


def add_grain(frame: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add per-channel Gaussian grain whose amplitude scales with luma.

    This gives the analyser a non-flat response curve to fit.
    """
    luma = (0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2])
    # Amplitude curve: low grain in shadows, peak in midtones, lower in highlights.
    amp = sigma * (0.4 + 0.6 * 4.0 * luma * (1.0 - luma))
    amp = amp[..., None].astype(np.float32)
    noise = rng.standard_normal(frame.shape).astype(np.float32) * amp
    return np.clip(frame + noise, 0.0, 1.0)


def to_uint16(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)


def write_prores(out_path: Path, frames: np.ndarray, fps: int) -> None:
    """Write a ProRes 4444 .mov from a (T, H, W, 3) uint16 array via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    t, h, w, c = frames.shape
    assert c == 3, f"expected RGB, got {c} channels"
    assert frames.dtype == np.uint16

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        # ffmpeg ingests rawvideo over stdin → re-encode to ProRes 4444.
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "prores_ks",
            "-profile:v", "4",          # 4444
            "-pix_fmt", "yuva444p10le",
            "-vendor", "apl0",
            "-bits_per_mb", "8000",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        try:
            assert proc.stdin is not None
            proc.stdin.write(frames.tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed (rc={rc}): {err}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_FIXTURES,
                   help="output directory for plate/degrained/comp .mov files")
    p.add_argument("--frames", type=int, default=12, help="number of frames per clip")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--sigma", type=float, default=0.05,
                   help="grain stddev at the peak of the response curve")
    p.add_argument("--seed", type=int, default=20260428)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Build frame stacks.
    clean_stack: list[np.ndarray] = []
    grainy_stack: list[np.ndarray] = []
    comp_stack: list[np.ndarray] = []
    for f in range(args.frames):
        t = f / max(1, args.frames - 1)
        clean = clean_frame(args.height, args.width, t)
        grainy = add_grain(clean, args.sigma, rng)
        comp = comp_recolour(clean)
        clean_stack.append(clean)
        grainy_stack.append(grainy)
        comp_stack.append(comp)

    clean_arr = to_uint16(np.stack(clean_stack))
    grainy_arr = to_uint16(np.stack(grainy_stack))
    comp_arr = to_uint16(np.stack(comp_stack))

    plate_path     = args.out / "plate.mov"
    degrained_path = args.out / "degrained.mov"
    comp_path      = args.out / "comp.mov"

    write_prores(plate_path,     grainy_arr, args.fps)
    write_prores(degrained_path, clean_arr,  args.fps)
    write_prores(comp_path,      comp_arr,   args.fps)

    # Also dump a single PNG preview of frame 0 for human inspection. PIL
    # / imageio's PNG writer wants 8-bit for RGB, so down-rez the previews.
    preview_dir = args.out / "preview"
    preview_dir.mkdir(exist_ok=True)
    def _u8(arr16: np.ndarray) -> np.ndarray:
        return (arr16.astype(np.uint32) >> 8).astype(np.uint8)
    iio.imwrite(preview_dir / "plate_f0.png",     _u8(grainy_arr[0]))
    iio.imwrite(preview_dir / "degrained_f0.png", _u8(clean_arr[0]))
    iio.imwrite(preview_dir / "comp_f0.png",      _u8(comp_arr[0]))

    plate_var = float(np.var(grainy_arr.astype(np.float32) / 65535.0))
    clean_var = float(np.var(clean_arr.astype(np.float32) / 65535.0))
    print(f"wrote {plate_path}")
    print(f"wrote {degrained_path}")
    print(f"wrote {comp_path}")
    print(f"plate variance = {plate_var:.5f}, degrained variance = {clean_var:.5f}")
    if plate_var <= clean_var:
        print("WARNING: plate variance not greater than degrained — "
              "grain injection may have failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
