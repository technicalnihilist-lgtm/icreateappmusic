"""Vocal / instrumental separation, locally, with numpy only.

Two modes:

  * "median"  - the classic repetition trick: instrumental parts repeat, the
                vocal does not, so a median across time per frequency bin
                estimates the backing track and the residual is the voice.
                Combined with a centre-channel (mid/side) mask, which works
                because lead vocals are almost always mixed dead centre.
  * "center"  - mid/side only. Instant, cruder, good enough to hear the words.

No models, no downloads. If you want studio-grade separation, install Demucs
(see README) and the app will use it automatically when available.
"""
import os
import subprocess
import wave

import numpy as np

SR = 44100
N_FFT = 2048
HOP = 512


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def _read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported wav width %d" % sw)
    if ch > 1:
        x = x.reshape(-1, ch).T
    else:
        x = x[None, :]
    return x, sr


def load_audio(path, sr=SR):
    """Load wav/mp3/m4a/ogg/flac -> (channels, samples) float32 at `sr`.

    mp3 and friends are decoded through ffmpeg when it is installed.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".wave"):
        x, in_sr = _read_wav(path)
    else:
        import shutil
        import tempfile
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "%s needs ffmpeg to decode. Install it (apt install ffmpeg / "
                "brew install ffmpeg) or convert the file to .wav first." % ext)
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-ac", "2", "-ar", str(sr), tmp], check=True)
        x, in_sr = _read_wav(tmp)
        try:
            os.remove(tmp)
        except OSError:
            pass
    if in_sr != sr:
        x = np.stack([resample(ch, in_sr, sr) for ch in x])
    if x.shape[0] > 2:
        x = x[:2]
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x, sr_in, sr_out):
    n_out = int(round(len(x) * sr_out / float(sr_in)))
    if n_out <= 0:
        return x
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


# --------------------------------------------------------------------------
# stft / istft
# --------------------------------------------------------------------------
def _hann(n):
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def stft(x, n_fft=N_FFT, hop=HOP):
    w = _hann(n_fft)
    n = x.shape[-1]
    frames = 1 + max(0, (n - n_fft) // hop)
    need = (frames - 1) * hop + n_fft - n
    if need > 0:
        xp = np.concatenate([x, np.zeros(need, dtype=np.float32)])
    else:
        xp = x
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    out = np.fft.rfft(xp[idx] * w, axis=1)
    return out.T  # (bins, frames)


def istft(S, n_fft=N_FFT, hop=HOP, length=None):
    w = _hann(n_fft)
    frames = S.shape[1]
    y = np.fft.irfft(S.T, n=n_fft, axis=1) * w
    total = (frames - 1) * hop + n_fft
    out = np.zeros(total, dtype=np.float32)
    norm = np.zeros(total, dtype=np.float32)
    for i in range(frames):
        a = i * hop
        out[a:a + n_fft] += y[i]
        norm[a:a + n_fft] += w * w
    out /= np.maximum(norm, 1e-8)
    if length:
        if out.size < length:
            out = np.concatenate([out, np.zeros(length - out.size, dtype=np.float32)])
        out = out[:length]
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _median_bg(mag, win=41, block=192):
    """Median of |STFT| across time, per bin, computed block-wise."""
    half = win // 2
    bins, frames = mag.shape
    padded = np.pad(mag, ((0, 0), (half, half)), mode="edge")
    out = np.empty_like(mag)
    view = np.lib.stride_tricks.sliding_window_view(padded, win, axis=1)
    for s in range(0, frames, block):
        e = min(frames, s + block)
        out[:, s:e] = np.median(view[:, s:e, :], axis=2)
    return out


def _center_ratio(M, S):
    a = np.abs(M)
    b = np.abs(S)
    return a / (a + b + 1e-9)


# --------------------------------------------------------------------------
# main separation
# --------------------------------------------------------------------------
def separate(path=None, audio=None, sr=SR, mode="center", win=215, strength=1.0,
             power=1.0, low_hz=None, high_hz=None, progress=None):
    """Split a mixed song into a vocal and a backing track.

    Returns dict with:
      vocals      (n,)   mono float32
      instrumental (2,n) stereo float32
      sr, mode, vocal_energy, music_energy
    """
    if audio is None:
        audio = load_audio(path, sr)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = np.vstack([x, x])
    n = x.shape[1]
    if progress:
        progress(0.15, "analysing the mix")

    mid = (x[0] + x[1]) * 0.5
    side = (x[0] - x[1]) * 0.5

    M = stft(mid)
    Ss = stft(side)
    if mode == "center":
        c = _center_ratio(M, Ss)
        mask = np.clip(c ** power, 0.0, 1.0)
        if progress:
            progress(0.6, "extracting the centre")
    else:
        if progress:
            progress(0.35, "estimating the backing track")
        mag = np.abs(M)
        bg = _median_bg(mag, win=win)
        if progress:
            progress(0.55, "masking the voice")
        resid = np.maximum(mag - 1.35 * bg, 0.0)
        mask = resid / (mag + 1e-9)
        mask = np.clip(mask, 0.0, 1.0) ** strength
        # centre emphasis: a voice panned to one side is almost never the lead
        mask *= (0.25 + 0.75 * _center_ratio(M, Ss))

    # vocals do not live below ~90 Hz or above ~12 kHz
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    lo = low_hz if low_hz is not None else (120.0 if mode == "center" else 90.0)
    hi = high_hz if high_hz is not None else (9000.0 if mode == "center" else 12000.0)
    band = np.ones_like(freqs)
    band[freqs < lo] = 0.0
    band[freqs > hi] = 0.0
    band = np.clip(band + (1.0 - band) * 0.05, 0.0, 1.0)
    mask = mask * band[:, None]

    if progress:
        progress(0.75, "rebuilding audio")
    vocals = istft(M * mask, length=n)
    music_mid = istft(M * (1.0 - mask), length=n)
    music = np.vstack([music_mid + side, music_mid - side]).astype(np.float32)

    # gentle cleanup
    vocals = vocals.astype(np.float32)
    peak = float(np.max(np.abs(vocals))) or 1.0
    if peak > 0.99:
        vocals *= 0.99 / peak

    if progress:
        progress(0.95, "done")
    return dict(vocals=vocals, instrumental=music, sr=sr, mode=mode,
                vocal_energy=float(np.sqrt(np.mean(vocals ** 2))),
                music_energy=float(np.sqrt(np.mean(music ** 2))))


def demucs_available():
    try:
        import torch  # noqa: F401
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def separate_demucs(path, sr=SR, progress=None):
    """Optional high-quality path: uses Demucs if it is installed."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if progress:
        progress(0.1, "loading Demucs")
    model = get_model("htdemucs")
    model.eval()
    x = load_audio(path, model.samplerate)
    wav = torch.from_numpy(x)[None]
    with torch.no_grad():
        out = apply_model(model, wav, split=True, progress=False)[0]
    names = model.sources
    stems = {k: out[i] for i, k in enumerate(names)}
    vocals = stems["vocals"].mean(0).numpy()
    music = sum(stems[k] for k in names if k != "vocals").numpy()
    if progress:
        progress(0.9, "done")
    if sr != model.samplerate:
        vocals = resample(vocals, model.samplerate, sr)
        music = np.stack([resample(c, model.samplerate, sr) for c in music])
    return dict(vocals=vocals.astype(np.float32),
                instrumental=music.astype(np.float32), sr=sr, mode="demucs",
                vocal_energy=float(np.sqrt(np.mean(vocals ** 2))),
                music_energy=float(np.sqrt(np.mean(music ** 2))))
