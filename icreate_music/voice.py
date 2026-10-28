"""Singing voice synthesis.

A formant synthesiser: a glottal pulse source (with vibrato, jitter and
breath noise) shaped by three resonant formants. Vowel formants plus
consonant noise bursts give an intelligible "sung" line. Any text can be
fed in; anything without lyrics is sung on vowels.

Every preset is generated from maths, so the app ships with no voice
samples and works with no network.
"""
import numpy as np

from .synth import (TAU, _t, lowpass, highpass, env_adsr, osc_additive,
                    saw_partials, midi_to_freq)

# F1, F2, F3 in Hz for a reference (adult male) vocal tract
VOWELS = {
    "a":  (730.0, 1090.0, 2440.0),
    "e":  (530.0, 1840.0, 2480.0),
    "i":  (270.0, 2290.0, 3010.0),
    "o":  (570.0,  840.0, 2410.0),
    "u":  (300.0,  870.0, 2240.0),
    "uh": (500.0, 1500.0, 2500.0),
    "ae": (660.0, 1720.0, 2410.0),
    "oh": (480.0,  900.0, 2400.0),
    "oo": (330.0,  780.0, 2200.0),
    "n":  (280.0, 1300.0, 2400.0),
}

VOICES = {
    # name: (formant_scale, vibrato_rate, vibrato_depth, breath, brightness, detune_voices)
    "male":        (1.00, 5.0, 0.0055, 0.10, 1.00, 1),
    "male_deep":   (0.88, 4.6, 0.0050, 0.12, 0.90, 1),
    "female":      (1.16, 5.6, 0.0065, 0.12, 1.05, 1),
    "female_high": (1.26, 5.9, 0.0070, 0.14, 1.12, 1),
    "child":       (1.34, 6.2, 0.0075, 0.10, 1.20, 1),
    "choir":       (1.10, 5.2, 0.0060, 0.16, 1.00, 3),
    "breathy":     (1.08, 5.4, 0.0075, 0.34, 1.02, 1),
    "robot":       (1.00, 0.0, 0.0000, 0.03, 1.10, 1),
    "whisper":     (1.06, 5.0, 0.0060, 0.85, 0.95, 1),
}

# Trained voice profiles, registered at runtime by train_voice.py
PROFILE_PARAMS = {}
PROFILE_LABELS = {}
PROFILE_META = {}


def register_profile(name, params):
    """Register a trained profile. Accepts a profile dict or a param tuple."""
    if isinstance(params, dict):
        tup = (float(params.get("formant_scale", 1.0)),
               float(params.get("vibrato_rate", 5.4)),
               float(params.get("vibrato_depth", 0.005)),
               float(params.get("breath", 0.10)),
               float(params.get("bright", 1.0)), 1)
        label = params.get("name") or name
        PROFILE_META[name] = params
    else:
        tup = tuple(params)
        label = name
    PROFILE_PARAMS[name] = tup
    PROFILE_LABELS[name] = label
    return tup


def resolve_voice(voice):
    """Built-in preset, trained profile, or fall back to female."""
    if voice in VOICES:
        return VOICES[voice]
    if voice in PROFILE_PARAMS:
        return PROFILE_PARAMS[voice]
    return VOICES["female"]


def all_voices():
    d = dict(VOICE_LABELS)
    for k, label in PROFILE_LABELS.items():
        d[k] = "Trained: %s" % label
    return d


VOICE_LABELS = {
    "male": "Male (warm)",
    "male_deep": "Male (deep)",
    "female": "Female (natural)",
    "female_high": "Female (bright)",
    "child": "Child-like",
    "choir": "Choir / stacked",
    "breathy": "Breathy (intimate)",
    "robot": "Robot / vocoder",
    "whisper": "Whisper",
}

VOWEL_LETTER = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "y": "i"}
DIGRAPHS = {"oo": "oo", "ee": "i", "ea": "i", "ai": "e", "ay": "e",
            "ou": "o", "ow": "o", "oa": "o", "ie": "i", "ei": "e",
            "au": "o", "aw": "o", "oi": "o", "uy": "i"}

PLOSIVES = set("ptkbdg")
FRICATIVES = set("sfhzvxjc")
NASALS = set("mnlrwy")


def _vowel_from_chunk(chunk):
    c = chunk.lower()
    for d, v in DIGRAPHS.items():
        if d in c:
            return v
    for ch in c:
        if ch in VOWEL_LETTER:
            return VOWEL_LETTER[ch]
    return "uh"


def _onset_from_chunk(chunk):
    c = chunk.lower().lstrip()
    if not c:
        return "none"
    if c.startswith(("sh", "ch", "th", "ph")):
        return "fricative"
    if c[0] in PLOSIVES:
        return "plosive"
    if c[0] in FRICATIVES:
        return "fricative"
    if c[0] in NASALS:
        return "nasal"
    return "none"


def syllabify(text):
    """Very light English-ish syllabifier.

    Returns a list of (onset_type, vowel) tuples, one per sung note. Words
    are split into vowel groups; a syllable with no vowel is dropped.
    """
    out = []
    for word in str(text).replace("-", " ").split():
        w = "".join(ch for ch in word.lower() if ch.isalpha() or ch == "'")
        if not w:
            continue
        groups = []
        cur = ""
        prev_vowel = False
        for ch in w:
            is_v = ch in VOWEL_LETTER
            if is_v and not prev_vowel:
                if cur:
                    groups.append(cur)
                cur = ch
            elif is_v and prev_vowel:
                cur += ch
            elif not is_v:
                cur += ch
            prev_vowel = is_v
        if cur:
            groups.append(cur)
        for g in groups:
            if any(ch in VOWEL_LETTER for ch in g):
                out.append((_onset_from_chunk(g), _vowel_from_chunk(g)))
    return out


def _formant_gain(sr, n, formants, scale=1.0, bright=1.0):
    f = np.fft.rfftfreq(n, 1.0 / sr)
    H = np.zeros_like(f)
    for i, (F, bw, amp) in enumerate(zip(formants, (90.0, 110.0, 150.0), (1.0, 0.55, 0.28))):
        F = F * scale
        H += amp / (1.0 + ((f - F) / (bw * scale)) ** 2)
    # singer's formant shimmer + gentle top end
    H += 0.10 / (1.0 + ((f - 2900.0 * scale) / (220.0 * scale)) ** 2) * bright
    H *= 1.0 / (1.0 + (f / (9000.0 * bright)) ** 2)
    return H


def _glottal(freq, n, sr, rng, bright=1.0, jitter=0.004):
    """Band-limited saw-ish glottal source with vibrato/jitter handled outside."""
    x = osc_additive(freq, n, sr, saw_partials(28, tilt=1.0))
    x += osc_additive(freq, n, sr, saw_partials(14, tilt=0.7)) * 0.35
    if jitter > 0:
        t = _t(n, sr)
        x *= 1.0 + jitter * np.sin(TAU * 3.7 * t) + jitter * 0.5 * np.sin(TAU * 11.3 * t)
    return x


def sing_note(freq, dur, sr, vowel="a", onset="none", voice="female",
              vel=1.0, rng=None, glide_from=None, legato=False):
    """Render one sung note. Returns mono float32."""
    rng = rng or np.random.default_rng(0)
    fscale, vrate, vdepth, breath, bright, stack = resolve_voice(voice)
    dur = max(0.05, float(dur))
    n = int((dur + 0.18) * sr)
    formants = VOWELS.get(vowel, VOWELS["uh"])

    # portamento from the previous pitch
    t = _t(n, sr)
    if glide_from and glide_from > 0 and glide_from != freq:
        glide = 0.045
        k = np.clip(t / glide, 0, 1)
        f_curve = glide_from * (freq / glide_from) ** (k ** 1.4)
    else:
        f_curve = np.full(n, float(freq))

    vib = 0.0
    if vrate > 0 and dur > 0.22:
        ramp = np.clip((t - 0.12) / 0.25, 0, 1)
        vib = vdepth * ramp * np.sin(TAU * vrate * t)
    f_curve = f_curve * (1.0 + vib)

    x = np.zeros(n)
    for s in range(stack):
        det = 1.0 + (s - (stack - 1) / 2.0) * 0.004
        ph = TAU * np.cumsum(f_curve * det) / sr + s * 1.7
        src = np.zeros(n)
        for k, (ratio, amp, _d) in enumerate(saw_partials(26, tilt=1.0)):
            if freq * det * ratio < sr * 0.45:
                src += amp * np.sin(ph * ratio + k * 0.4)
        x += src / max(1, stack)

    # breath / aspiration noise
    if breath > 0:
        noise = rng.normal(0, 1, n)
        noise = highpass(noise, sr, 900)
        noise = lowpass(noise, sr, 7000)
        x = (1.0 - breath * 0.6) * x + breath * noise * 1.4

    # vowel formants
    nfft = 1 << int(np.ceil(np.log2(max(2, n))))
    X = np.fft.rfft(x, nfft)
    H = _formant_gain(sr, nfft, formants, scale=fscale, bright=bright)
    x = np.fft.irfft(X * H, nfft)[:n]
    x = highpass(x.astype(np.float32), sr, 85)

    if voice == "robot":
        x = x * (0.65 + 0.35 * np.sin(TAU * 42.0 * t))
        x = np.round(x * 24) / 24.0

    # consonant onset
    if onset != "none" and not legato:
        cl = int(min(0.09, max(0.02, 0.035)) * sr)
        if onset == "plosive":
            seg = rng.normal(0, 1, cl) * np.exp(-np.arange(cl) / (cl * 0.25))
            seg = lowpass(seg, sr, 3200) * 0.30
        elif onset == "fricative":
            seg = rng.normal(0, 1, cl) * np.hanning(cl)
            seg = highpass(lowpass(seg, sr, 8000), sr, 2400) * 0.22
        else:  # nasal
            seg = rng.normal(0, 1, cl) * np.hanning(cl)
            seg = lowpass(seg, sr, 1200) * 0.14
        x[:cl] += seg * (1.0 if onset != "fricative" else 1.2)

    env = env_adsr(n, sr, a=0.02 if not legato else 0.008,
                   d=0.10, s=0.92, r=0.14 if legato else 0.10)
    x = x * env * vel
    return x.astype(np.float32)


def sing_phrase(notes, sr, lyrics=None, voice="female", rng=None, legato=True):
    """Render a monophonic sung phrase.

    notes: list of dicts with freq, dur, start (seconds) -- returned as a
    single mono buffer with the notes placed at their start times.
    """
    rng = rng or np.random.default_rng(0)
    if not notes:
        return np.zeros(0, dtype=np.float32)
    end = max(nt["start"] + nt["dur"] for nt in notes) + 0.4
    out = np.zeros(int(end * sr) + 1, dtype=np.float32)
    syls = syllabify(lyrics) if lyrics else []
    prev_f = None
    for i, nt in enumerate(notes):
        if syls:
            onset, vowel = syls[i % len(syls)]
        else:
            vowel, onset = ("a", "none")
        gap = None
        if i + 1 < len(notes):
            gap = notes[i + 1]["start"] - (nt["start"] + nt["dur"])
        this_legato = legato and gap is not None and gap < 0.06
        buf = sing_note(nt["freq"], nt["dur"], sr, vowel=vowel, onset=onset,
                        voice=voice, vel=nt.get("vel", 1.0), rng=rng,
                        glide_from=prev_f, legato=this_legato)
        s = int(max(0.0, nt["start"]) * sr)
        if s >= out.size:
            continue
        e = min(out.size, s + buf.size)
        if e <= s:
            continue
        out[s:e] += buf[:e - s]
        prev_f = nt["freq"]
    return out
