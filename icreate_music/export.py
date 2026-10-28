"""Writing audio and metadata to disk. stdlib only (wave module)."""
import json
import os
import wave

import numpy as np


def _to_int16(x, dither=True):
    x = np.asarray(x, dtype=np.float64)
    if dither:
        rng = np.random.default_rng(12345)
        x = x + (rng.random(x.shape) - rng.random(x.shape)) / 32768.0
    return np.clip(x, -1.0, 1.0)


def write_wav(path, audio, sr, bits=16):
    """audio: (N,) mono or (2, N) stereo float in [-1, 1]."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    n_ch = a.shape[0]
    n = a.shape[1]
    a = _to_int16(a)
    data = np.ascontiguousarray(a.T)  # interleaved
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(n_ch)
        w.setsampwidth(bits // 8)
        w.setframerate(int(sr))
        if bits == 16:
            w.writeframes((data * 32767.0).astype("<i2").tobytes())
        elif bits == 24:
            ints = (data * 8388607.0).astype("<i4")
            raw = ints.view(np.uint8).reshape(-1, 4)[:, :3]
            w.writeframes(raw.tobytes())
        else:
            raise ValueError("bits must be 16 or 24")
    return path


def write_mp3(path, wav_path, bitrate="192k"):
    """Optional: needs ffmpeg on PATH. Returns path or None."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return None
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                        "-codec:a", "libmp3lame", "-b:a", bitrate, path],
                       check=True)
        return path
    except Exception:
        return None


def write_stems(out_dir, stems, sr):
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for name, buf in stems.items():
        p = os.path.join(out_dir, "%s.wav" % name)
        write_wav(p, buf, sr)
        made.append(p)
    return made


def write_manifest(path, info):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return path


def safe_name(text):
    keep = "".join(c if (c.isalnum() or c in " -_") else "" for c in str(text))
    return "-".join(keep.split())[:60] or "song"
