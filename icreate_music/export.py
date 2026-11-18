"""Writing audio and metadata to disk. stdlib only (wave module)."""
import json
import os
import unicodedata
import wave

import numpy as np


# frames converted per pass when writing a wav: 64k stereo frames is ~256 kB
# of scratch, small enough for a phone and large enough that the per-block
# overhead disappears.
_WRITE_BLOCK = 1 << 16


def _to_int16(x, dither=True, rng=None, levels=32768.0):
    """Clip to [-1, 1] and add TPDF dither, in float32.

    float64 here bought nothing: the 16-bit output cannot resolve the
    difference, and it doubled the memory of the whole file.

    `levels` is the dither step, 2**(bits-1). It used to be fixed at the
    16-bit step, so a 24-bit export carried 256 LSBs of noise and never held
    more than 16 real bits.
    """
    x = np.asarray(x, dtype=np.float32)
    if dither:
        rng = rng if rng is not None else np.random.default_rng(12345)
        x = x + (rng.random(x.shape, dtype=np.float32)
                 - rng.random(x.shape, dtype=np.float32)) / levels
    return np.clip(x, -1.0, 1.0)


def write_wav(path, audio, sr, bits=16):
    """audio: (N,) mono or (2, N) stereo float in [-1, 1].

    Converted one block at a time. Converting the whole file at once held
    several full-length float64 copies of it -- about 600 MB for a five-minute
    stereo song, which is an out-of-memory crash on a phone, and a 5 s stall
    even on a desktop.
    """
    if bits not in (16, 24):
        # checked before the file is opened: the old order left a header-only
        # wav behind every time an odd bit depth was asked for
        raise ValueError("bits must be 16 or 24")
    a = np.asarray(audio)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("audio must be mono (N,) or stereo (2, N)")
    n_ch = int(a.shape[0])
    n = int(a.shape[1])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    rng = np.random.default_rng(12345)
    scale = 32767.0 if bits == 16 else 8388607.0
    levels = 32768.0 if bits == 16 else 8388608.0
    with wave.open(path, "wb") as w:
        w.setnchannels(n_ch)
        w.setsampwidth(bits // 8)
        w.setframerate(int(sr))
        for s in range(0, n, _WRITE_BLOCK):
            blk = _to_int16(a[:, s:s + _WRITE_BLOCK], rng=rng, levels=levels)
            data = np.ascontiguousarray(blk.T)      # interleaved
            if bits == 16:
                raw = (data * scale).astype("<i2")
            else:
                # float64 only for the 24-bit multiply: in float32 the low
                # bits of a 24-bit value do not survive the scaling
                ints = (data.astype(np.float64) * scale).astype("<i4")
                raw = ints.view(np.uint8).reshape(-1, 4)[:, :3]
            w.writeframes(raw.tobytes())
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


def _name_char(c):
    """Letters, digits and vowel signs survive; punctuation does not.

    Combining marks have to be kept explicitly: dropping them turns
    "\u0906\u0938\u092e\u093e\u0928" into "\u0906\u0938\u092e\u0928", which is a different word.
    """
    return (c.isalnum() or c in " -_"
            or unicodedata.category(c).startswith("M"))


def safe_name(text):
    keep = "".join(c if _name_char(c) else "" for c in str(text))
    return "-".join(keep.split())[:60] or "song"
