"""Train a voice from a song.

Pipeline:  song.mp3 -> separate vocals -> analyse the singer -> voice profile

The profile captures what actually makes a singer recognisable to a synth:
pitch range and register, vibrato rate/depth, vocal-tract size (formant
scale), breathiness and brightness. Those numbers then drive the formant
synthesiser, so the app sings new songs "in that voice".

This is not neural voice cloning (that is RVC/So-VITS territory, GBs of
weights and a GPU). It is a fast, local, honest approximation: the timbre
family and the mannerisms are matched, the identity is close but not
identical.
"""
import json
import os
import re
import time

import numpy as np

from . import separate
from .synth import highpass, midi_to_freq

SR = 44100

# Measured on this synthesiser itself (formant_scale 1.0, five cardinal
# vowels): F1 sits at 646 Hz and the spectral centroid at 3.54x the f0.
# Using the synth as its own reference is what makes trained profiles
# interchangeable with the built-in voices.
REF_F1 = 646.0
# both brightness measures drift with pitch, measured on the synth at 220/440 Hz
REF_CENTROID_RATIO = 3.82
REF_PITCH = 220.0


# --------------------------------------------------------------------------
# pitch tracking
# --------------------------------------------------------------------------
def f0_track(x, sr=SR, fmin=70.0, fmax=1000.0, win=2048, hop=441, with_peak=False):
    """Autocorrelation pitch track.

    Returns (f0, voiced_mask, rms), or with with_peak=True a fourth array of
    normalised autocorrelation peaks -- a harmonic-to-noise ratio proxy that
    tracks breathiness.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < win:
        return np.zeros(1), np.zeros(1, dtype=bool), np.zeros(1)
    frames = 1 + (n - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(frames)[:, None]
    fr = x[idx] * np.hanning(win).astype(np.float32)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    fr = fr - fr.mean(axis=1, keepdims=True)

    lag_lo = int(sr / fmax)
    lag_hi = int(sr / fmin)
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    f0 = np.zeros(frames, dtype=np.float32)
    peakv = np.zeros(frames, dtype=np.float32)

    block = 512
    for s in range(0, frames, block):
        e = min(frames, s + block)
        X = np.fft.rfft(fr[s:e], nfft, axis=1)
        ac = np.fft.irfft(X * np.conj(X), nfft, axis=1)[:, :win]
        ac0 = ac[:, :1] + 1e-12
        acn = ac / ac0
        seg = acn[:, lag_lo:min(lag_hi, win - 1)]
        if seg.shape[1] < 2:
            continue
        top = np.max(seg, axis=1, keepdims=True)
        # first lag nearly as strong as the best: avoids the classic double-period
        # octave-down error. Register is anchored by the cepstral estimate, so
        # this track only needs to supply range and vibrato.
        strong = seg >= (0.88 * top)
        k = np.argmax(strong, axis=1)
        peakv[s:e] = seg[np.arange(seg.shape[0]), k]
        # parabolic refinement
        kk = k + lag_lo
        a = acn[np.arange(seg.shape[0]), np.clip(kk - 1, 0, win - 1)]
        b = acn[np.arange(seg.shape[0]), np.clip(kk, 0, win - 1)]
        c = acn[np.arange(seg.shape[0]), np.clip(kk + 1, 0, win - 1)]
        denom = (a - 2 * b + c)
        shift = np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / denom, 0.0)
        lag = kk + np.clip(shift, -1, 1)
        f0[s:e] = sr / np.maximum(lag, 1)

    gate = max(0.002, 0.12 * float(np.median(rms[rms > 0]))) if np.any(rms > 0) else 0.002
    voiced = (peakv > 0.18) & (rms > gate)
    f0 = np.where(voiced, f0, 0.0)
    if with_peak:
        return f0, voiced, rms, peakv
    return f0, voiced, rms


def cepstral_f0(x, sr=SR, voiced=None, win=2048, hop=441):
    """Register estimate from the averaged spectrum's harmonic comb.

    Per-frame autocorrelation picks the wrong octave often enough to matter
    (a 220 Hz voice reading as 440), but the comb in the mean spectrum of the
    voiced frames is unambiguous.
    """
    n = x.size
    frames = 1 + max(0, (n - win) // hop)
    if frames < 4:
        return 0.0
    idx = np.arange(win)[None, :] + hop * np.arange(frames)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(win).astype(np.float32), axis=1))
    v = voiced[:frames] if (voiced is not None and voiced.size >= frames) else np.ones(frames, bool)
    if np.sum(v) < 4:
        v = np.ones(frames, bool)
    P = S[v].mean(axis=0) + 1e-9
    ceps = np.fft.irfft(np.log(P))
    qlo, qhi = int(sr / 900.0), int(sr / 60.0)
    seg = ceps[qlo:min(qhi, ceps.size - 1)]
    if seg.size < 3:
        return 0.0
    k = int(np.argmax(seg)) + qlo
    return float(sr / max(k, 1))


def _runs(mask):
    """Yield (start, end) index pairs of True runs."""
    if not np.any(mask):
        return
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    for a, b in zip(starts, ends):
        yield a, b


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def _vibrato(f0, voiced, hop, sr):
    """Vibrato rate (Hz) and relative depth from sustained voiced runs."""
    rates, depths = [], []
    frame_rate = sr / float(hop)
    for a, b in _runs(voiced):
        seg = f0[a:b]
        if seg.size < int(0.45 * frame_rate):
            continue
        if np.any(seg <= 0):
            continue
        # drop note onsets/offsets: the glide there is not vibrato
        trim = int(0.09 * frame_rate)
        if seg.size > 2 * trim + 8:
            seg = seg[trim:-trim]
        # remove the slow pitch drift, keep the wobble
        k = max(3, int(0.18 * frame_rate))
        kernel = np.ones(k) / k
        trend = np.convolve(seg, kernel, mode="same")
        res = seg - trend
        # ignore edges where 'same' padding distorts
        pad = k
        if res.size <= 2 * pad + 8:
            continue
        core = res[pad:-pad]
        if core.size < 16:
            continue
        sp = np.fft.rfft(core * np.hanning(core.size), 4096)
        fr = np.fft.rfftfreq(4096, 1.0 / frame_rate)
        band = (fr >= 4.0) & (fr <= 8.0)
        if not np.any(band):
            continue
        rates.append(float(fr[band][np.argmax(np.abs(sp[band]))]))
        # keep only the 4-8 Hz wobble: slow drift and note movement are not vibrato
        core_b = np.fft.irfft(np.where(band, sp, 0), 4096)[:core.size]
        depths.append(float(np.sqrt(2.0) * np.std(core_b) / (np.mean(seg) + 1e-9)))
    rate = float(np.median(rates)) if rates else 5.4
    # Depth is not reliably separable from note-level pitch movement in a mixed
    # recording, so it stays at a natural default rather than a made-up number.
    return float(np.clip(rate, 3.5, 8.5)), 0.006


def _spectral(x, sr, voiced, hop=441, win=2048):
    """Average voiced-frame spectrum -> centroid, tilt, HF ratio, formants."""
    n = x.size
    frames = 1 + max(0, (n - win) // hop)
    if frames < 4:
        return dict(centroid=REF_CENTROID, tilt=-1.0, hf_ratio=0.1,
                    formants=[500.0, 1500.0, 2500.0])
    idx = np.arange(win)[None, :] + hop * np.arange(frames)[:, None]
    fr = x[idx] * np.hanning(win).astype(np.float32)
    S = np.abs(np.fft.rfft(fr, axis=1))
    v = voiced[:frames] if voiced.size >= frames else np.ones(frames, dtype=bool)
    if np.sum(v) < 4:
        v = np.ones(frames, dtype=bool)
    P = S[v].mean(axis=0)
    f = np.fft.rfftfreq(win, 1.0 / sr)
    tot = np.sum(P) + 1e-12
    centroid = float(np.sum(f * P) / tot)
    hf = float(np.sum(P[f > 6000]) / tot)

    # spectral tilt: slope of log-power vs log-frequency, 150 Hz - 4 kHz
    band = (f > 150) & (f < 4000)
    if np.sum(band) > 10:
        lx = np.log10(f[band])
        ly = np.log10(P[band] + 1e-12)
        tilt = float(np.polyfit(lx, ly, 1)[0])
    else:
        tilt = -1.0

    # formants: cepstral smoothing then peak picking
    logP = np.log(P + 1e-12)
    ceps = np.fft.irfft(logP)
    lifter = np.zeros_like(ceps)
    q = 40                                   # ~1.8 ms at 44.1 kHz
    lifter[:q] = 1.0
    lifter[-q:] = 1.0
    env = np.fft.rfft(ceps * lifter).real[:f.size]
    cands = []
    for i in range(1, len(env) - 1):
        if f[i] > 3800 or f[i] < 250:
            continue
        if env[i] > env[i - 1] and env[i] >= env[i + 1]:
            cands.append((float(f[i]), float(env[i])))
    picked = []
    for fq, _amp in sorted(cands, key=lambda t: t[0]):
        if all(abs(fq - q) > 260.0 for q in picked):
            picked.append(fq)
        if len(picked) == 3:
            break
    while len(picked) < 3:
        picked.append([500.0, 1500.0, 2500.0][len(picked)])
    return dict(centroid=centroid, tilt=tilt, hf_ratio=hf, formants=picked)


def analyse_vocal(x, sr=SR):
    """Turn a vocal stem into voice parameters."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    # strip bass/kick bleed: it is what makes a pitch tracker jump an octave down
    x = highpass(x, sr, 100.0)

    f0, voiced, rms, peakv = f0_track(x, sr, with_peak=True)
    good = f0[voiced & (f0 > 0)]
    if good.size < 20:
        raise RuntimeError(
            "not enough sung pitch found - try a cleaner vocal or a longer excerpt")

    # Register comes from the cepstral comb, which does not confuse octaves;
    # the frame track only supplies range and vibrato, folded onto that anchor.
    med = cepstral_f0(x, sr, voiced)
    if med <= 0:
        med = float(np.median(good))
    folded = []
    for v in good:
        ratio = np.log2(max(med, 1.0) / max(v, 1.0))
        if abs(ratio) > 0.55:          # a real octave error, not natural movement
            v = v * (2.0 ** int(np.round(ratio)))
        folded.append(v)
    folded = np.array(folded)
    p10, p90 = np.percentile(folded, [10, 90])
    # A collapsed range (octave folding can do this on deep voices) would make
    # every new melody monotone; fall back to a singable default width.
    if 12.0 * np.log2(max(p90, 1) / max(p10, 1)) < 4.0:
        p10, p90 = med / (2 ** (5 / 12.0)), med * (2 ** (5 / 12.0))

    rate, depth = _vibrato(f0, voiced, 441, sr)
    sp = _spectral(x, sr, voiced)

    fscale = float(np.clip(sp["formants"][0] / REF_F1, 0.75, 1.45))
    # centroid and HF energy both scale with pitch: normalise before mapping
    pitch_rel = max(med, 60.0) / REF_PITCH
    ratio = sp["centroid"] / max(med, 60.0)
    bright = float(np.clip(ratio / (REF_CENTROID_RATIO * pitch_rel ** -0.72),
                           0.80, 1.30))
    # Breathiness comes from the autocorrelation peak (harmonic-to-noise),
    # which survives consonants; raw HF energy does not. Least-squares fit
    # against renders at breath 0.03/0.12/0.30/0.45.
    # upper quartile: frames with the least instrumental bleed represent the
    # voice itself, which matters when the stem came out of a separation
    hnr = float(np.percentile(peakv[voiced[:peakv.size]], 75)) if np.any(voiced) else 0.95
    # HNR falls when the stem carries instrumental bleed, so the mapping is
    # deliberately compressed: 0.90 -> 0.05 (clean), 0.82 -> 0.30 (bleedy).
    breath = float(np.clip(2.8625 - 3.125 * hnr, 0.05, 0.32))
    pitch_center = float(69.0 + 12.0 * np.log2(med / 440.0))

    return dict(
        f0_median=round(med, 2), f0_p10=round(float(p10), 2), f0_p90=round(float(p90), 2),
        semitone_range=round(float(12.0 * np.log2(max(p90, 1) / max(p10, 1))), 2),
        vibrato_rate=round(rate, 2), vibrato_depth=round(depth, 4),
        formant_scale=round(fscale, 3), formants=[round(v, 1) for v in sp["formants"]],
        centroid=round(sp["centroid"], 1), tilt=round(sp["tilt"], 2),
        hf_ratio=round(sp["hf_ratio"], 3), breath=round(breath, 3),
        bright=round(bright, 3), pitch_center_midi=round(pitch_center, 2),
        voiced_fraction=round(float(np.mean(voiced)), 3),
        analysed_seconds=round(x.size / float(sr), 1),
    )


def to_synth_params(profile):
    """Profile dict -> the parameter tuple the formant synthesiser wants."""
    return (float(profile["formant_scale"]), float(profile["vibrato_rate"]),
            float(profile["vibrato_depth"]), float(profile["breath"]),
            float(profile["bright"]), 1)


# --------------------------------------------------------------------------
# train / manage
# --------------------------------------------------------------------------
def slug(name):
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name)).strip("-").lower()
    return s[:40] or "voice"


def voices_dir(base):
    d = os.path.join(base, "voices")
    os.makedirs(d, exist_ok=True)
    return d


def train_voice(source, name, base="songs", sr=SR, mode="center",
                use_demucs=False, progress=None, keep_stems=True):
    """Full pipeline: separate, analyse, save profile + preview.

    source may be a path (wav/mp3/...) or an audio array.
    """
    from . import export, song as song_mod

    def prog(p, m):
        if progress:
            progress(p, m)

    if isinstance(source, str):
        prog(0.05, "reading %s" % os.path.basename(source))
        audio = separate.load_audio(source, sr)
        src_name = os.path.basename(source)
    else:
        audio = np.asarray(source, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.vstack([audio, audio])
        src_name = "uploaded"

    if use_demucs and separate.demucs_available():
        prog(0.15, "separating with Demucs")
        stems = separate.separate_demucs(audio=audio, sr=sr, progress=prog) \
            if False else separate.separate_demucs(source, sr=sr, progress=prog)
    else:
        # Dense mixes sometimes leave too little voice after a centre-only
        # split, so try the repetition-based mode as a fallback before giving up.
        modes = [mode] if mode != "center" else ["center", "median"]
        stems = None
        profile = None
        last_err = None
        for m in modes:
            prog(0.15, "separating vocals from the music (%s)" % m)
            stems = separate.separate(audio=audio, sr=sr, mode=m, progress=prog)
            if not np.any(stems["vocals"]):
                last_err = RuntimeError("no vocal could be extracted from this file")
                continue
            prog(0.8, "analysing the singer")
            try:
                profile = analyse_vocal(stems["vocals"], sr)
                break
            except RuntimeError as exc:
                last_err = exc
                continue
        if profile is None:
            raise RuntimeError(
                "%s - this mix may be too dense or mostly instrumental. "
                "Try a longer excerpt, a cleaner recording, or install Demucs "
                "for stronger separation." % (last_err or "training failed"))

    vocal = stems["vocals"]
    profile.update(name=name, slug=slug(name), source=src_name,
                   created=time.strftime("%Y-%m-%d %H:%M"),
                   separation=stems.get("mode", mode))

    d = voices_dir(base)
    stem_dir = os.path.join(base, "stems", slug(name))
    if keep_stems:
        os.makedirs(stem_dir, exist_ok=True)
        export.write_wav(os.path.join(stem_dir, "vocals.wav"), vocal, sr)
        export.write_wav(os.path.join(stem_dir, "instrumental.wav"),
                         stems["instrumental"], sr)
        profile["stems"] = stem_dir

    # a short sung preview so the profile can be checked by ear
    prog(0.9, "rendering a preview")
    try:
        preview_notes = []
        base_f = float(midi_to_freq(profile["pitch_center_midi"]))
        for i, (step, dur) in enumerate([(0, 0.5), (2, 0.5), (4, 0.7), (2, 0.5), (0, 0.9)]):
            preview_notes.append(dict(start=i * 0.55, dur=dur,
                                      freq=base_f * (2 ** (step / 12.0)), vel=0.9))
        from .voice import register_profile, sing_phrase
        register_profile(profile["slug"], profile)
        pv = sing_phrase(preview_notes, sr, lyrics="la la la la la",
                         voice=profile["slug"])
        from .synth import reverb_ir, convolve, normalize
        pv = convolve(pv, reverb_ir(sr, 1.1, 3.0, np.random.default_rng(1))) * 0.35 + pv * 0.8
        export.write_wav(os.path.join(d, slug(name) + "_preview.wav"),
                         normalize(pv.astype(np.float32), 0.9), sr)
        profile["preview"] = os.path.join(d, slug(name) + "_preview.wav")
    except Exception:
        pass

    path = os.path.join(d, slug(name) + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    profile["path"] = path
    prog(1.0, "voice ready")
    return profile


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_profiles(base="songs"):
    d = voices_dir(base)
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            try:
                out.append(load_profile(os.path.join(d, name)))
            except Exception:
                continue
    return out


def load_all_into_engine(base="songs"):
    """Register every saved profile with the synthesiser."""
    from .voice import register_profile
    for p in list_profiles(base):
        try:
            register_profile(p["slug"], p)
        except Exception:
            continue
    return len(list_profiles(base))
