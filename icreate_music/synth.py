"""Synthesis primitives: oscillators, instruments, drums and effects.

Pure numpy. No samples, no downloads, no network: every sound is generated
from mathematics at render time.
"""
import numpy as np

TAU = 2.0 * np.pi


# --------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------
def midi_to_freq(m):
    return 440.0 * (2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0))


def _t(n, sr):
    return np.arange(n, dtype=np.float64) / sr


def osc_additive(freq, n, sr, partials, vib_rate=0.0, vib_depth=0.0, phase0=0.0):
    """Sum of sine partials with per-partial exponential decay.

    partials: list of (ratio, amplitude, decay_per_second)
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = _t(n, sr)
    if vib_depth > 0:
        f = freq * (1.0 + vib_depth * np.sin(TAU * vib_rate * t))
    else:
        f = np.full(n, float(freq))
    ph = TAU * np.cumsum(f) / sr + phase0
    out = np.zeros(n)
    nyq = sr * 0.45
    for ratio, amp, decay in partials:
        if freq * ratio >= nyq or amp == 0.0:
            continue
        if decay > 0:
            out += amp * np.exp(-t * decay) * np.sin(ph * ratio)
        else:
            out += amp * np.sin(ph * ratio)
    return out


def saw_partials(n_harm, rolloff=1.0, decay_scale=0.0, tilt=1.0):
    return [(float(k), (1.0 / k ** tilt) * rolloff, decay_scale * k) for k in range(1, n_harm + 1)]


def square_partials(n_harm, rolloff=1.0):
    return [(float(k), (1.0 / k) * rolloff, 0.0) for k in range(1, n_harm + 1, 2)]


def env_adsr(n, sr, a=0.01, d=0.1, s=0.7, r=0.15):
    a_n = max(1, int(a * sr))
    d_n = max(1, int(d * sr))
    r_n = max(1, int(r * sr))
    if a_n + d_n + r_n > n:
        k = n / float(a_n + d_n + r_n)
        a_n = max(1, int(a_n * k))
        d_n = max(1, int(d_n * k))
        r_n = max(1, n - a_n - d_n)
    s_n = max(0, n - a_n - d_n - r_n)
    parts = [np.linspace(0.0, 1.0, a_n, endpoint=False),
             np.linspace(1.0, s, d_n, endpoint=False),
             np.full(s_n, s),
             np.linspace(s, 0.0, r_n)]
    e = np.concatenate(parts)
    if e.size < n:
        e = np.concatenate([e, np.zeros(n - e.size)])
    return e[:n].astype(np.float32)


def env_perc(n, sr, attack=0.004, decay=6.0):
    t = _t(n, sr)
    e = (1.0 - np.exp(-t / max(attack, 1e-5))) * np.exp(-t * decay)
    m = np.max(np.abs(e)) or 1.0
    return (e / m).astype(np.float32)


# --------------------------------------------------------------------------
# filters (FFT based: exact, vectorised, no scipy)
# --------------------------------------------------------------------------
def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(2, n))))


def _fft_shape(x, sr, response):
    n = x.shape[-1]
    if n < 8:
        return x.astype(np.float32)
    # power-of-two FFT: avoids Bluestein's algorithm on awkward prime sizes
    nfft = _next_pow2(n)
    X = np.fft.rfft(x.astype(np.float64), nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    y = np.fft.irfft(X * response(f), nfft)
    return y[..., :n].astype(np.float32)


def lowpass(x, sr, cutoff, order=2):
    cutoff = max(20.0, min(cutoff, sr * 0.45))
    return _fft_shape(x, sr, lambda f: 1.0 / np.sqrt(1.0 + (f / cutoff) ** (2 * order)))


def highpass(x, sr, cutoff, order=2):
    cutoff = max(10.0, min(cutoff, sr * 0.45))
    return _fft_shape(x, sr, lambda f: 1.0 / np.sqrt(1.0 + (cutoff / np.maximum(f, 1e-6)) ** (2 * order)))


def bandpass(x, sr, lo, hi, order=2):
    return lowpass(highpass(x, sr, lo, order), sr, hi, order)


def resonant_sweep(x, sr, f_start, f_end, q=2.0):
    """Cheap formant-style sweep, used for wobbles and risers."""
    n = x.shape[-1]
    if n < 8:
        return x.astype(np.float32)
    nfft = _next_pow2(n)
    X = np.fft.rfft(x.astype(np.float64), nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    centre = f_start + (f_end - f_start) * 0.5
    bw = max(60.0, centre / q)
    H = 1.0 / (1.0 + ((f - centre) / bw) ** 2)
    return np.fft.irfft(X * H, nfft)[:n].astype(np.float32)


# --------------------------------------------------------------------------
# physical-ish pluck (Karplus-Strong, block vectorised)
# --------------------------------------------------------------------------
def _karplus(freq, n, sr, ring=1.6, bright=0.5, damp=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    P = int(max(2, round(sr / max(freq, 20.0))))
    if n <= P + 2:
        return np.zeros(n, dtype=np.float32)
    b = np.zeros(n + P + 2, dtype=np.float64)
    exc = rng.uniform(-1.0, 1.0, P)
    # soften the excitation: brightness 0 = dark/thumpy, 1 = bright/twangy
    if bright < 0.99:
        k = 1.0 - bright
        exc = np.convolve(exc, np.ones(max(1, int(1 + k * 6))) / max(1, int(1 + k * 6)), mode="same")
    b[1:P + 1] = exc
    decay = 10.0 ** (-3.0 / max(ring, 0.05) / sr)
    a = decay * damp
    a2 = decay * (1.0 - damp)
    i = P + 1
    while i < n + 1:
        j = min(i + P, n + 1)
        b[i:j] = a * b[i - P:j - P] + a2 * b[i - P - 1:j - P - 1]
        i = j
    out = b[1:n + 1]
    out *= np.linspace(1.0, 0.0, n) ** 0.25  # gentle final fade, kills loop residue
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# instruments.  Each returns mono float32 of length dur + tail.
# --------------------------------------------------------------------------
def inst_piano(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.55) * sr)
    parts = []
    for k in range(1, 16):
        ratio = k * (1.0 + 0.00035 * k * k)          # string inharmonicity
        parts.append((ratio, (1.0 / k ** 1.35), 1.1 + 0.85 * k))
    x = osc_additive(freq, n, sr, parts, phase0=0.0)
    # hammer noise
    rng = rng or np.random.default_rng(1)
    click = rng.normal(0, 1, min(n, int(0.02 * sr)))
    x[:click.size] += lowpass(click, sr, 3000) * 0.08
    e = env_perc(n, sr, attack=0.002, decay=1.4)
    body = env_perc(n, sr, attack=0.002, decay=0.55)
    return (x * (0.45 * e + 0.75 * body) * vel).astype(np.float32)


def inst_epiano(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.5) * sr)
    parts = [(1.0, 1.0, 1.0), (2.0, 0.32, 2.1), (3.01, 0.13, 3.2), (4.0, 0.06, 4.5), (7.0, 0.03, 7.0)]
    x = osc_additive(freq, n, sr, parts, vib_rate=4.6, vib_depth=0.0025)
    t = _t(n, sr)
    x *= (1.0 + 0.12 * np.sin(TAU * 4.6 * t))
    return (x * env_perc(n, sr, 0.004, 1.1) * vel * 0.9).astype(np.float32)


def inst_guitar(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.35) * sr)
    x = _karplus(freq, n, sr, ring=1.8, bright=0.62, damp=0.5, rng=rng)
    x = lowpass(x, sr, 5200)
    e = env_adsr(n, sr, a=0.002, d=0.05, s=1.0, r=0.12)
    return (x * e * vel).astype(np.float32)


def inst_pluck(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.3) * sr)
    x = _karplus(freq, n, sr, ring=0.7, bright=0.85, damp=0.52, rng=rng)
    return (lowpass(x, sr, 7000) * vel).astype(np.float32)


def inst_pad(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.9) * sr)
    x = np.zeros(n)
    for det in (-0.006, 0.0, 0.007):
        x += osc_additive(freq * (1 + det), n, sr, saw_partials(9, tilt=1.15),
                          vib_rate=0.13, vib_depth=0.0018)
    x = lowpass(x, sr, min(2600, freq * 12 + 600))
    e = env_adsr(n, sr, a=0.55, d=0.5, s=0.82, r=0.85)
    return (x * e * vel * 0.22).astype(np.float32)


def inst_strings(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.7) * sr)
    x = np.zeros(n)
    for det in (-0.005, 0.002, 0.006):
        x += osc_additive(freq * (1 + det), n, sr, saw_partials(11, tilt=1.1),
                          vib_rate=5.1, vib_depth=0.005)
    x = lowpass(x, sr, 3800)
    x = highpass(x, sr, 120)
    e = env_adsr(n, sr, a=0.22, d=0.3, s=0.85, r=0.55)
    return (x * e * vel * 0.2).astype(np.float32)


def inst_brass(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.4) * sr)
    x = osc_additive(freq, n, sr, saw_partials(18, tilt=0.95), vib_rate=5.4, vib_depth=0.004)
    x = lowpass(x, sr, 2400)
    e = env_adsr(n, sr, a=0.06, d=0.25, s=0.8, r=0.3)
    return (x * e * vel * 0.3).astype(np.float32)


def inst_lead(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.3) * sr)
    x = osc_additive(freq, n, sr, saw_partials(12, tilt=1.0), vib_rate=5.6, vib_depth=0.006)
    x += osc_additive(freq, n, sr, square_partials(9), vib_rate=5.6, vib_depth=0.006) * 0.4
    x = lowpass(x, sr, 5200)
    e = env_adsr(n, sr, a=0.012, d=0.12, s=0.75, r=0.2)
    return (x * e * vel * 0.26).astype(np.float32)


def inst_organ(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.25) * sr)
    bars = [(1, 1.0), (2, 0.55), (3, 0.38), (4, 0.3), (6, 0.18), (8, 0.13)]
    x = osc_additive(freq, n, sr, [(r, a, 0.0) for r, a in bars])
    t = _t(n, sr)
    x *= 1.0 + 0.08 * np.sin(TAU * 5.8 * t)
    e = env_adsr(n, sr, a=0.02, d=0.05, s=0.9, r=0.18)
    return (x * e * vel * 0.2).astype(np.float32)


def inst_bass(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.18) * sr)
    x = osc_additive(freq, n, sr, [(1.0, 1.0, 0.0)] + [(float(k), 0.5 / k, 0.0) for k in range(2, 9)])
    x += osc_additive(freq, n, sr, [(1.0, 0.5, 0.0)], phase0=np.pi / 2) * 0.3
    x = lowpass(x, sr, 420 + 500 * vel)
    e = env_adsr(n, sr, a=0.006, d=0.14, s=0.62, r=0.09)
    return (x * e * vel * 0.85).astype(np.float32)


def inst_sub(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.2) * sr)
    x = osc_additive(freq, n, sr, [(1.0, 1.0, 0.0), (2.0, 0.12, 0.0)])
    e = env_adsr(n, sr, a=0.01, d=0.2, s=0.8, r=0.12)
    return (x * e * vel * 0.8).astype(np.float32)


def inst_bell(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 1.6) * sr)
    parts = [(1.0, 1.0, 0.9), (2.01, 0.5, 1.5), (2.98, 0.35, 2.1),
             (4.16, 0.22, 2.8), (5.43, 0.14, 3.6), (6.79, 0.09, 4.5)]
    x = osc_additive(freq, n, sr, parts)
    return (x * env_perc(n, sr, 0.002, 1.6) * vel * 0.6).astype(np.float32)


def inst_marimba(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.4) * sr)
    parts = [(1.0, 1.0, 3.2), (4.02, 0.28, 6.0), (9.1, 0.08, 9.0)]
    x = osc_additive(freq, n, sr, parts)
    return (x * env_perc(n, sr, 0.003, 3.0) * vel * 0.7).astype(np.float32)


def inst_flute(freq, dur, sr, vel=1.0, rng=None):
    n = int((dur + 0.25) * sr)
    x = osc_additive(freq, n, sr, [(1.0, 1.0, 0.0), (2.0, 0.08, 0.0)], vib_rate=5.2, vib_depth=0.004)
    rng = rng or np.random.default_rng(3)
    breath = lowpass(rng.normal(0, 1, n), sr, 4500) * 0.06
    e = env_adsr(n, sr, a=0.05, d=0.1, s=0.85, r=0.2)
    return ((x + breath) * e * vel * 0.5).astype(np.float32)


INSTRUMENTS = {
    "piano": inst_piano,
    "epiano": inst_epiano,
    "guitar": inst_guitar,
    "pluck": inst_pluck,
    "pad": inst_pad,
    "strings": inst_strings,
    "brass": inst_brass,
    "lead": inst_lead,
    "organ": inst_organ,
    "bass": inst_bass,
    "sub": inst_sub,
    "bell": inst_bell,
    "marimba": inst_marimba,
    "flute": inst_flute,
}


# --------------------------------------------------------------------------
# drums
# --------------------------------------------------------------------------
def dr_kick(sr, vel=1.0, rng=None, tight=1.0):
    n = int(0.5 * sr)
    t = _t(n, sr)
    f = 45.0 + 105.0 * np.exp(-t * 34.0)
    ph = TAU * np.cumsum(f) / sr
    body = np.sin(ph) * np.exp(-t * 6.5 * tight)
    click = np.exp(-t * 220.0) * 0.35 * np.sin(TAU * 1250 * t)
    x = body + click
    return (x * env_perc(n, sr, 0.001, 0.4) * vel).astype(np.float32)


def dr_snare(sr, vel=1.0, rng=None):
    rng = rng or np.random.default_rng(5)
    n = int(0.35 * sr)
    t = _t(n, sr)
    noise = rng.normal(0, 1, n)
    noise = bandpass(noise, sr, 900, 7200)
    tone = np.sin(TAU * 186 * t) * 0.5 + np.sin(TAU * 279 * t) * 0.25
    x = noise * np.exp(-t * 22.0) + tone * np.exp(-t * 26.0)
    return (x * env_perc(n, sr, 0.001, 0.2) * vel * 0.9).astype(np.float32)


def dr_rim(sr, vel=1.0, rng=None):
    rng = rng or np.random.default_rng(6)
    n = int(0.09 * sr)
    t = _t(n, sr)
    x = bandpass(rng.normal(0, 1, n), sr, 1400, 4200) * np.exp(-t * 70.0)
    return (x * vel * 0.8).astype(np.float32)


def dr_hat(sr, vel=1.0, rng=None, open_=False):
    rng = rng or np.random.default_rng(7)
    n = int((0.42 if open_ else 0.09) * sr)
    t = _t(n, sr)
    noise = rng.normal(0, 1, n)
    x = highpass(noise, sr, 7200)
    x = x * np.exp(-t * (10.0 if open_ else 62.0))
    return (x * vel * 0.5).astype(np.float32)


def dr_clap(sr, vel=1.0, rng=None):
    rng = rng or np.random.default_rng(8)
    n = int(0.3 * sr)
    t = _t(n, sr)
    noise = bandpass(rng.normal(0, 1, n), sr, 900, 5200)
    env = np.zeros(n)
    for off in (0.0, 0.011, 0.023):
        s = int(off * sr)
        seg = np.exp(-(t[:n - s]) * 55.0)
        env[s:] += seg
    env += np.exp(-t * 12.0) * 0.6
    return (noise * env * vel * 0.55).astype(np.float32)


def dr_tom(sr, vel=1.0, rng=None, base=140.0):
    n = int(0.4 * sr)
    t = _t(n, sr)
    f = base * (1.0 + 0.7 * np.exp(-t * 22.0))
    x = np.sin(TAU * np.cumsum(f) / sr) * np.exp(-t * 9.0)
    return (x * vel * 0.85).astype(np.float32)


def dr_crash(sr, vel=1.0, rng=None):
    rng = rng or np.random.default_rng(9)
    n = int(1.8 * sr)
    t = _t(n, sr)
    x = highpass(rng.normal(0, 1, n), sr, 4200)
    x = x * np.exp(-t * 2.6)
    return (x * vel * 0.35).astype(np.float32)


def dr_shaker(sr, vel=1.0, rng=None):
    rng = rng or np.random.default_rng(10)
    n = int(0.12 * sr)
    t = _t(n, sr)
    x = bandpass(rng.normal(0, 1, n), sr, 3200, 9000)
    return (x * np.exp(-t * 45.0) * vel * 0.35).astype(np.float32)


DRUMS = {
    "kick": lambda sr, vel, rng: dr_kick(sr, vel, rng),
    "snare": lambda sr, vel, rng: dr_snare(sr, vel, rng),
    "rim": lambda sr, vel, rng: dr_rim(sr, vel, rng),
    "hat": lambda sr, vel, rng: dr_hat(sr, vel, rng, False),
    "hatopen": lambda sr, vel, rng: dr_hat(sr, vel, rng, True),
    "clap": lambda sr, vel, rng: dr_clap(sr, vel, rng),
    "tom": lambda sr, vel, rng: dr_tom(sr, vel, rng),
    "tomlow": lambda sr, vel, rng: dr_tom(sr, vel, rng, 95.0),
    "crash": lambda sr, vel, rng: dr_crash(sr, vel, rng),
    "shaker": lambda sr, vel, rng: dr_shaker(sr, vel, rng),
}


# --------------------------------------------------------------------------
# effects
# --------------------------------------------------------------------------
def pan_gains(pan):
    """pan in [-1, 1] -> (left, right) equal-power gains."""
    p = (np.clip(pan, -1, 1) + 1.0) * np.pi / 4.0
    return float(np.cos(p)), float(np.sin(p))


def reverb_ir(sr, seconds=1.9, decay=3.2, rng=None, damp=5200):
    rng = rng or np.random.default_rng(11)
    n = int(seconds * sr)
    t = _t(n, sr)
    ir = rng.normal(0, 1, n) * np.exp(-t * decay)
    ir = lowpass(ir, sr, damp)
    ir[:int(0.008 * sr)] *= np.linspace(0.2, 1.0, int(0.008 * sr))
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    return ir.astype(np.float32)


def convolve(x, h, block=None):
    """Overlap-add FFT convolution, memory friendly."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    m = h.size
    out = np.zeros(n + m - 1, dtype=np.float32)
    if block is None:
        block = 1 << int(np.ceil(np.log2(max(m * 3, 16384))))
    step = max(1, block - m + 1)
    pos = 0
    nfft = 1 << int(np.ceil(np.log2(block + m)))
    H = np.fft.rfft(h.astype(np.float64), nfft)
    while pos < n:
        seg = x[pos:pos + step]
        X = np.fft.rfft(seg.astype(np.float64), nfft)
        y = np.fft.irfft(X * H, nfft)
        end = min(pos + seg.size + m - 1, out.size)
        out[pos:end] += y[:end - pos].astype(np.float32)
        pos += step
    return out[:n]


def chorus(x, sr, depth_ms=8.0, rate=0.35, mix=0.35):
    n = x.size
    if n < 64:
        return x
    t = _t(n, sr)
    d = (depth_ms * 0.001 * sr) * (1.0 + np.sin(TAU * rate * t))
    idx = np.arange(n) - d
    idx = np.clip(idx, 0, n - 1)
    wet = np.interp(idx, np.arange(n), x.astype(np.float64))
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def delay(x, sr, time_s=0.33, feedback=0.34, mix=0.25, pingpong=False):
    n = x.size
    d = int(time_s * sr)
    if d <= 0 or n < d:
        return x
    out = x.astype(np.float32).copy()
    tap = x.astype(np.float32).copy()
    g = feedback
    for _ in range(6):
        shifted = np.zeros(n, dtype=np.float32)
        shifted[d:] = tap[:-d] * g
        out += shifted * mix
        tap = shifted
        g *= feedback
    return out


def compress(x, sr, thresh_db=-16.0, ratio=2.5, smooth_ms=80.0):
    """Simple, fast, frame-based downward compressor. Stereo-safe (per channel)."""
    x = np.asarray(x, dtype=np.float32)
    mono = x.ndim == 1
    sig = x if mono else x.mean(axis=0)
    frame = max(64, int(sr * 0.005))
    nf = max(1, sig.size // frame)
    frames = sig[:nf * frame].reshape(nf, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)
    over = np.maximum(0.0, db - thresh_db)
    gain_db = -over * (1.0 - 1.0 / ratio)
    # smooth the gain curve
    k = np.exp(-1.0 / max(1.0, (smooth_ms * 0.001 * sr) / frame))
    sm = np.empty_like(gain_db)
    acc = 0.0
    for i in range(gain_db.size):
        acc = acc * k + gain_db[i] * (1 - k)
        sm[i] = acc
    gain = np.repeat(10.0 ** (sm / 20.0), frame)
    if gain.size < sig.size:
        gain = np.concatenate([gain, np.full(sig.size - gain.size, gain[-1] if gain.size else 1.0)])
    if mono:
        return (x * gain[:x.size]).astype(np.float32)
    return (x * gain[None, :x.shape[1]]).astype(np.float32)


def soft_clip(x, drive=1.0):
    return (np.tanh(x * drive) / max(np.tanh(drive), 1e-6)).astype(np.float32)


def normalize(x, peak=0.95):
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-9:
        return x.astype(np.float32)
    return (x * (peak / m)).astype(np.float32)


def fade(x, sr, fin=0.01, fout=0.05):
    x = x.copy()
    a = min(x.shape[-1], int(fin * sr))
    b = min(x.shape[-1], int(fout * sr))
    if a > 0:
        ramp = np.linspace(0, 1, a) ** 0.6
        x[..., :a] *= ramp
    if b > 0:
        ramp = np.linspace(1, 0, b) ** 0.6
        x[..., -b:] *= ramp
    return x
