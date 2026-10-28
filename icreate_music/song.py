"""The composer: turns a seed + a few choices into a finished stereo song.

Everything here is deterministic given the seed, so the same seed always
rebuilds the same song. Nothing is downloaded and nothing is sampled: all
audio is synthesised at render time.
"""
import numpy as np

from . import theory
from .synth import (INSTRUMENTS, DRUMS, midi_to_freq, pan_gains, reverb_ir,
                    convolve, chorus, delay, compress, soft_clip, normalize,
                    fade, lowpass, highpass, env_adsr)
from .voice import sing_phrase, VOICES, VOICE_LABELS

# --------------------------------------------------------------------------
# genres
# --------------------------------------------------------------------------
# arch: (section name, bars, energy 0..1)
GENRES = {
    "lofi": dict(
        bpm=(72, 88), scales=["dorian", "minor", "major"], swing=0.18,
        keys="epiano", pad="pad", bass="bass", arp="pluck", lead=None,
        drums=dict(kick="x.......x.x.....", snare="....x.......x...",
                   hat="x.x.x.x.x.x.x.x.", shaker="..x...x...x...x."),
        reverb=2.1, vocal=True, arp_on=True, pump=0.0,
        arch=[("intro", 4, 0.30), ("verse", 8, 0.55), ("chorus", 8, 0.78),
              ("verse", 8, 0.58), ("chorus", 8, 0.82), ("outro", 4, 0.34)],
        mix=dict(keys=(0.52, -0.18, 0.40), pad=(0.30, 0.22, 0.60),
                 bass=(0.62, 0.0, 0.10), arp=(0.26, 0.35, 0.55),
                 lead=(0.30, 0.10, 0.35), vocal=(0.78, 0.0, 0.32),
                 drums=(0.72, 0.0, 0.16)),
    ),
    "pop": dict(
        bpm=(104, 124), scales=["major", "minor"], swing=0.0,
        keys="piano", pad="strings", bass="bass", arp="pluck", lead="bell",
        drums=dict(kick="x.......x.......", snare="....x.......x...",
                   hat="..x...x...x...x.", clap="....x.......x..."),
        reverb=1.7, vocal=True, arp_on=False, pump=0.0,
        arch=[("intro", 4, 0.35), ("verse", 8, 0.60), ("chorus", 8, 0.92),
              ("verse", 8, 0.62), ("chorus", 8, 0.96), ("bridge", 4, 0.50),
              ("chorus", 8, 1.0), ("outro", 4, 0.40)],
        mix=dict(keys=(0.50, -0.22, 0.35), pad=(0.26, 0.28, 0.55),
                 bass=(0.66, 0.0, 0.08), arp=(0.22, 0.38, 0.50),
                 lead=(0.28, 0.15, 0.40), vocal=(0.86, 0.0, 0.28),
                 drums=(0.80, 0.0, 0.14)),
    ),
    "edm": dict(
        bpm=(122, 132), scales=["minor", "harmonic_minor"], swing=0.0,
        keys="pad", pad="pad", bass="sub", arp="pluck", lead="lead",
        drums=dict(kick="x...x...x...x...", clap="....x.......x...",
                   hat="..x...x...x...x.", hatopen="......x.......x."),
        reverb=1.9, vocal=True, arp_on=True, pump=0.55,
        arch=[("intro", 8, 0.30), ("verse", 8, 0.62), ("chorus", 8, 1.0),
              ("verse", 8, 0.66), ("chorus", 8, 1.0), ("outro", 8, 0.35)],
        mix=dict(keys=(0.34, 0.0, 0.50), pad=(0.30, 0.0, 0.55),
                 bass=(0.72, 0.0, 0.05), arp=(0.24, -0.30, 0.55),
                 lead=(0.34, 0.0, 0.40), vocal=(0.72, 0.0, 0.35),
                 drums=(0.88, 0.0, 0.12)),
    ),
    "synthwave": dict(
        bpm=(96, 112), scales=["minor", "dorian"], swing=0.0,
        keys="pad", pad="strings", bass="bass", arp="pluck", lead="lead",
        drums=dict(kick="x...x...x...x...", snare="....x.......x...",
                   hat="x.x.x.x.x.x.x.x.", tom="..............x."),
        reverb=2.4, vocal=True, arp_on=True, pump=0.0,
        arch=[("intro", 8, 0.32), ("verse", 8, 0.60), ("chorus", 8, 0.90),
              ("verse", 8, 0.64), ("chorus", 8, 0.94), ("outro", 8, 0.36)],
        mix=dict(keys=(0.36, -0.20, 0.55), pad=(0.30, 0.25, 0.62),
                 bass=(0.66, 0.0, 0.06), arp=(0.26, 0.30, 0.55),
                 lead=(0.30, -0.12, 0.45), vocal=(0.74, 0.0, 0.34),
                 drums=(0.82, 0.0, 0.16)),
    ),
    "rock": dict(
        bpm=(112, 140), scales=["minor", "major", "mixolydian"], swing=0.0,
        keys="organ", pad=None, bass="bass", arp=None, lead="brass",
        drums=dict(kick="x.....x.x.......", snare="....x.......x...",
                   hat="x.x.x.x.x.x.x.x.", crash="x..............."),
        reverb=1.3, vocal=True, arp_on=False, pump=0.0,
        arch=[("intro", 4, 0.42), ("verse", 8, 0.68), ("chorus", 8, 0.95),
              ("verse", 8, 0.70), ("chorus", 8, 0.98), ("outro", 4, 0.45)],
        mix=dict(keys=(0.34, -0.24, 0.28), pad=(0.0, 0.0, 0.0),
                 bass=(0.70, 0.0, 0.06), arp=(0.0, 0.0, 0.0),
                 lead=(0.34, 0.18, 0.35), vocal=(0.82, 0.0, 0.26),
                 drums=(0.86, 0.0, 0.12)),
    ),
    "hiphop": dict(
        bpm=(84, 96), scales=["minor", "minor_pentatonic", "blues"], swing=0.12,
        keys="epiano", pad="pad", bass="bass", arp="pluck", lead=None,
        drums=dict(kick="x..x..x...x.....", snare="....x.......x...",
                   hat="x.xxx.xxx.xxx.xx", rim="..............x."),
        reverb=1.6, vocal=True, arp_on=False, pump=0.0,
        arch=[("intro", 4, 0.32), ("verse", 8, 0.62), ("chorus", 8, 0.85),
              ("verse", 8, 0.64), ("chorus", 8, 0.88), ("outro", 4, 0.34)],
        mix=dict(keys=(0.50, -0.15, 0.35), pad=(0.26, 0.20, 0.55),
                 bass=(0.74, 0.0, 0.05), arp=(0.20, 0.30, 0.50),
                 lead=(0.26, 0.10, 0.35), vocal=(0.84, 0.0, 0.30),
                 drums=(0.84, 0.0, 0.14)),
    ),
    "cinematic": dict(
        bpm=(64, 84), scales=["minor", "harmonic_minor", "dorian"], swing=0.0,
        keys="piano", pad="strings", bass="sub", arp="bell", lead="strings",
        drums=dict(kick="x.......x.......", tom="........x.x.....", tom_low=None),
        reverb=3.0, vocal=False, arp_on=True, pump=0.0,
        arch=[("intro", 8, 0.20), ("verse", 8, 0.45), ("chorus", 8, 0.80),
              ("bridge", 8, 0.55), ("chorus", 8, 0.95), ("outro", 8, 0.30)],
        mix=dict(keys=(0.46, -0.20, 0.55), pad=(0.40, 0.22, 0.70),
                 bass=(0.60, 0.0, 0.10), arp=(0.24, 0.32, 0.65),
                 lead=(0.34, 0.0, 0.55), vocal=(0.60, 0.0, 0.45),
                 drums=(0.70, 0.0, 0.25)),
    ),
    "ambient": dict(
        bpm=(60, 76), scales=["lydian", "major", "dorian"], swing=0.0,
        keys="pad", pad="strings", bass="sub", arp="bell", lead="flute",
        drums=dict(shaker="..x...x...x...x."),
        reverb=4.0, vocal=False, arp_on=True, pump=0.0,
        arch=[("intro", 8, 0.18), ("verse", 8, 0.40), ("chorus", 8, 0.62),
              ("verse", 8, 0.44), ("outro", 8, 0.24)],
        mix=dict(keys=(0.44, -0.25, 0.75), pad=(0.42, 0.25, 0.80),
                 bass=(0.44, 0.0, 0.25), arp=(0.22, 0.35, 0.80),
                 lead=(0.28, -0.10, 0.70), vocal=(0.50, 0.0, 0.60),
                 drums=(0.30, 0.0, 0.45)),
    ),
    "folk": dict(
        bpm=(88, 106), scales=["major", "mixolydian", "dorian"], swing=0.06,
        keys="guitar", pad=None, bass="bass", arp="guitar", lead="flute",
        drums=dict(kick="x.......x.......", snare="....x.......x...",
                   shaker="..x...x...x...x."),
        reverb=1.5, vocal=True, arp_on=True, pump=0.0,
        arch=[("intro", 4, 0.32), ("verse", 8, 0.58), ("chorus", 8, 0.80),
              ("verse", 8, 0.60), ("chorus", 8, 0.84), ("outro", 4, 0.36)],
        mix=dict(keys=(0.56, -0.22, 0.30), pad=(0.0, 0.0, 0.0),
                 bass=(0.52, 0.0, 0.08), arp=(0.34, 0.28, 0.35),
                 lead=(0.26, 0.12, 0.40), vocal=(0.86, 0.0, 0.26),
                 drums=(0.56, 0.0, 0.18)),
    ),
    "jazz": dict(
        bpm=(96, 132), scales=["dorian", "mixolydian", "major"], swing=0.26,
        keys="epiano", pad=None, bass="bass", arp=None, lead="brass",
        drums=dict(kick="x.......x.......", rim="..x...x...x...x.",
                   hat="..x...x...x...x.", snare="....x.......x..."),
        reverb=1.8, vocal=True, arp_on=False, pump=0.0,
        arch=[("intro", 4, 0.40), ("verse", 8, 0.65), ("chorus", 8, 0.85),
              ("verse", 8, 0.68), ("chorus", 8, 0.88), ("outro", 4, 0.42)],
        mix=dict(keys=(0.50, -0.20, 0.35), pad=(0.0, 0.0, 0.0),
                 bass=(0.62, 0.0, 0.10), arp=(0.0, 0.0, 0.0),
                 lead=(0.32, 0.15, 0.40), vocal=(0.80, 0.0, 0.30),
                 drums=(0.66, 0.0, 0.20)),
    ),
}

GENRE_LABELS = {
    "lofi": "Lo-fi / chill", "pop": "Pop", "edm": "EDM / dance",
    "synthwave": "Synthwave", "rock": "Rock", "hiphop": "Hip-hop / trap",
    "cinematic": "Cinematic / score", "ambient": "Ambient",
    "folk": "Folk / acoustic", "jazz": "Jazz",
}

TRACK_ORDER = ["drums", "bass", "keys", "pad", "arp", "lead", "vocal"]

# per-track loudness targets (RMS, linear). Mix gains then act as faders,
# so every genre and seed lands in the same ballpark balance.
TRACK_LEVEL = {"drums": 0.105, "bass": 0.115, "keys": 0.075, "pad": 0.060,
               "arp": 0.045, "lead": 0.062, "vocal": 0.115}

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _add(buf, s, x, gain=1.0):
    """Mix buffer `x` into `buf` at sample offset `s`, clamped safely."""
    if x.size == 0:
        return
    if s < 0:
        x = x[-s:]
        s = 0
    if s >= buf.size or x.size == 0:
        return
    en = min(buf.size, s + x.size)
    if en <= s:
        return
    buf[s:en] += x[:en - s] * gain


def _steps(pattern):
    return [(i, ch) for i, ch in enumerate(pattern) if ch in "xXoO"]


def _vel_of(ch):
    return 1.0 if ch in "xX" else 0.62


def _swing(beat, swing, rng):
    """Nudge off-grid notes for a human feel."""
    if swing <= 0:
        return beat
    step = beat * 4.0
    frac = step - np.floor(step)
    if abs(frac) < 0.01:
        i = int(round(step))
        if i % 4 == 2:
            return beat + swing * 0.16
        if i % 2 == 1:
            return beat + swing * 0.07
    return beat


def build_arrangement(genre, target_bars):
    arch = GENRES[genre]["arch"]
    out = []
    total = 0
    i = 0
    while total < target_bars and i < 64:
        name, bars, energy = arch[i % len(arch)]
        if total + bars > target_bars:
            bars = target_bars - total
        if bars <= 0:
            break
        out.append({"name": name, "bars": bars, "energy": energy})
        total += bars
        i += 1
    return out


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------
def _rhythm_templates(density, rng):
    pool = [
        [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
        [(0, 1.5), (1.5, 0.5), (2, 1.0), (3, 1.0)],
        [(0, 1.0), (1, 0.5), (1.5, 0.5), (2, 1.0), (3, 1.0)],
        [(0, 0.5), (0.5, 0.5), (1, 1.0), (2, 1.5), (3.5, 0.5)],
        [(0, 2.0), (2, 1.0), (3, 1.0)],
        [(0, 1.0), (2, 1.0), (3, 0.5), (3.5, 0.5)],
        [(0, 0.5), (0.5, 0.5), (1, 0.5), (1.5, 0.5), (2, 1.0), (3, 1.0)],
        [(0, 1.5), (2, 0.5), (2.5, 1.5)],
    ]
    if density < 0.5:
        pool = pool[1:6]
    return pool[int(rng.integers(0, len(pool)))]


def _gen_melody(bars, chords, scale_pitches, rng, density, lo_idx, hi_idx, register_center):
    """Motif-based melody. Returns list of (start_beat, dur_beats, midi)."""
    notes = []
    motif = None
    centre = theory.nearest_scale_index(scale_pitches, register_center)
    idx = max(lo_idx, min(hi_idx, centre))
    for bar in range(bars):
        if motif is not None and bar % 4 in (1, 2, 3):
            base = motif
            if bar % 4 == 3 and rng.random() < 0.5:
                shift = rng.choice([-2, -1, 1, 2])
                base = [(s, d, i + shift) for s, d, i in motif]
            for s, d, i in base:
                if rng.random() < density * 0.85:
                    notes.append((bar * 4 + s, d, i))
            continue
        if rng.random() < (1.0 - density) * 0.35 and bar > 0:
            continue  # rest bar
        rhythm = _rhythm_templates(density, rng)
        local = []
        chord = chords[bar % len(chords)]
        for s, d in rhythm:
            strong = abs(s - round(s)) < 0.01 and round(s) % 2 == 0
            if rng.random() > density:
                if not strong:
                    continue
            move = int(rng.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3]))
            # mean reversion: keep the line arching around the register centre
            # instead of drifting up against the ceiling
            pull = (centre - idx) * 0.28
            idx = int(round(max(lo_idx, min(hi_idx, idx + move + pull))))
            if strong:
                tones = [theory.nearest_scale_index(scale_pitches, p) for p in chord]
                tones = [t for t in tones if lo_idx <= t <= hi_idx]
                if tones:
                    idx = min(tones, key=lambda t: abs(t - idx))
            local.append((s, d, idx))
        motif = local
        for s, d, i in local:
            notes.append((bar * 4 + s, d, i))
    return [(s, d, scale_pitches[i]) for s, d, i in notes]


def _gen_bass(bars, chords, rng, style, energy):
    out = []
    for bar in range(bars):
        chord = chords[bar % len(chords)]
        root = min(chord, key=lambda p: p)
        while root > 48:
            root -= 12
        while root < 33:
            root += 12
        fifth = root + 7
        if style == "walk":
            pat = [(0, 1.0, root), (1, 0.9, root + rng.choice([2, 4, 5, 7])),
                   (2, 1.0, fifth), (3, 0.9, root + rng.choice([9, 10, 11, 12]))]
        elif style == "drive":
            pat = [(i * 0.5, 0.45, root if i % 4 != 3 else root + 12) for i in range(8)]
        elif style == "pulse":
            pat = [(0, 1.4, root), (2, 0.9, root), (3, 0.9, fifth)]
        elif style == "sub":
            pat = [(0, 3.6, root)]
        else:  # root-fifth
            pat = [(0, 1.6, root), (2, 1.4, root), (3, 0.9, fifth)]
        for s, d, p in pat:
            if s >= 3.0 and energy < 0.45 and rng.random() < 0.5:
                continue
            out.append((bar * 4 + s, d, p))
    return out


def _gen_chords(bars, chords, rng, style, energy):
    """Returns (start, dur, [midi...]) blocks."""
    out = []
    for bar in range(bars):
        chord = chords[bar % len(chords)]
        if style == "hold":
            out.append((bar * 4, 4.0, chord))
        elif style == "stab":
            for s in (0, 1.5, 2.5):
                out.append((bar * 4 + s, 0.9, chord))
        elif style == "offbeat":
            for s in (0.5, 1.5, 2.5, 3.5):
                out.append((bar * 4 + s, 0.42, chord))
        elif style == "arp":
            tones = chord + [chord[0] + 12]
            seq = [0, 1, 2, 3, 2, 1, 2, 3]
            for i, k in enumerate(seq):
                out.append((bar * 4 + i * 0.5, 0.46, [tones[k % len(tones)]]))
        else:
            out.append((bar * 4, 2.0, chord))
            out.append((bar * 4 + 2, 2.0, chord))
    return out


# --------------------------------------------------------------------------
# main render
# --------------------------------------------------------------------------
DEFAULT_SR = 44100


def render(params, progress=None):
    """Render a song. Returns (stereo float32 array, info dict, stems dict)."""
    sr = int(params.get("sr", DEFAULT_SR))
    seed = int(params.get("seed") or theory.random_seed())
    rng = np.random.default_rng(seed)
    genre = params.get("genre", "pop")
    if genre not in GENRES:
        genre = "pop"
    g = GENRES[genre]

    bpm = int(params.get("bpm") or rng.integers(g["bpm"][0], g["bpm"][1] + 1))
    root = int(params.get("key", -1))
    if root < 0:
        root = int(rng.integers(0, 12))
    scale = params.get("scale") or str(rng.choice(g["scales"]))
    spb = 60.0 / bpm

    seconds = float(params.get("seconds", 90))
    target_bars = max(4, int(round(seconds / (4.0 * spb))))
    sections = build_arrangement(genre, target_bars)

    voice = params.get("voice", "female")
    if voice not in VOICES:
        voice = "female"
    want_vocal = bool(params.get("vocal", g["vocal"]))
    lyrics = (params.get("lyrics") or "").strip()
    voice_octave = int(params.get("voice_octave", 0))

    # instruments can be overridden
    keys_inst = params.get("keys_inst") or g["keys"]
    pad_inst = params.get("pad_inst") or g["pad"]
    bass_inst = params.get("bass_inst") or g["bass"]
    arp_inst = params.get("arp_inst") or g["arp"]
    lead_inst = params.get("lead_inst") or g["lead"]

    progression = theory.pick_progression(scale, rng)
    scale_pitches = theory.scale_pitches(root, scale, octave=4)

    # chord per bar over the whole song
    all_chords = []
    for sec_i, sec in enumerate(sections):
        for bar in range(sec["bars"]):
            roman = progression[(bar // 2) % len(progression)]
            octv = 3 if sec["energy"] < 0.5 else 4
            all_chords.append(theory.chord_pitches(root, scale, roman, octave=octv))

    total_bars = sum(s["bars"] for s in sections)
    total_samples = int(total_bars * 4 * spb * sr) + int(3.0 * sr)

    # ---- compose every track's events first (cheap) --------------------
    tracks = {k: [] for k in TRACK_ORDER}
    vocal_notes = []
    bar_offset = 0
    motif_seed_state = rng.bit_generator.state
    for sec in sections:
        bars = sec["bars"]
        energy = sec["energy"]
        name = sec["name"]
        chords = all_chords[bar_offset:bar_offset + bars]
        if not chords:
            break
        # melody register
        centre = 69 if voice in ("female", "female_high", "child") else 57
        centre += 12 * voice_octave
        lo, hi = centre - 9, centre + 10
        mel_density = 0.42 + 0.5 * energy
        if name == "intro":
            mel_density *= 0.5
        mel = _gen_melody(bars, chords, scale_pitches, rng, mel_density,
                          theory.nearest_scale_index(scale_pitches, lo),
                          theory.nearest_scale_index(scale_pitches, hi), centre)
        for s, d, midi in mel:
            t = bar_offset * 4 + s
            tracks["vocal"].append(dict(start=t, dur=d * 0.92, midi=midi,
                                        vel=0.75 + 0.2 * energy))
            if lead_inst:
                tracks["lead"].append(dict(start=t, dur=d * 0.9, midi=midi + 12,
                                           vel=0.6 + 0.3 * energy))
        # bass
        bstyle = "walk" if genre == "jazz" else ("drive" if genre == "edm" else
                                                 ("sub" if genre == "ambient" else "root"))
        if name == "intro" and genre in ("pop", "rock"):
            bstyle = "root"
        for s, d, midi in _gen_bass(bars, chords, rng, bstyle, energy):
            tracks["bass"].append(dict(start=bar_offset * 4 + s, dur=d,
                                       midi=midi, vel=0.7 + 0.25 * energy))
        # keys
        kstyle = "hold" if name in ("intro", "outro") and genre in ("ambient", "cinematic") else (
            "offbeat" if genre in ("edm", "synthwave") else
            ("stab" if genre in ("pop", "rock", "jazz") else "hold"))
        for s, d, ch in _gen_chords(bars, chords, rng, kstyle, energy):
            for i, midi in enumerate(ch):
                tracks["keys"].append(dict(start=bar_offset * 4 + s, dur=d,
                                           midi=midi + 12, vel=0.55 + 0.3 * energy))
        # pad
        if pad_inst:
            for s, d, ch in _gen_chords(bars, chords, rng, "hold", energy):
                for midi in ch:
                    tracks["pad"].append(dict(start=bar_offset * 4 + s, dur=d * 1.05,
                                              midi=midi, vel=0.45 + 0.35 * energy))
        # arp
        if arp_inst and g["arp_on"] and energy > 0.3:
            for s, d, ch in _gen_chords(bars, chords, rng, "arp", energy):
                for midi in ch:
                    tracks["arp"].append(dict(start=bar_offset * 4 + s, dur=d,
                                              midi=midi + 12,
                                              vel=0.35 + 0.35 * energy))
        # drums
        if name != "intro" or genre in ("rock", "edm"):
            for drum, pattern in g["drums"].items():
                if pattern is None:
                    continue
                for step, ch in _steps(pattern):
                    vel = _vel_of(ch) * (0.55 + 0.45 * energy)
                    if drum in ("hat", "shaker") and step % 4 != 0:
                        vel *= 0.72
                    if energy < 0.4 and drum in ("snare", "clap") and step % 8 != 4:
                        continue
                    tracks["drums"].append(dict(
                        start=bar_offset * 4 + step * 0.25, dur=0.25,
                        drum=drum, vel=vel))
            # fill at the end of each 4-bar group
            if bars >= 4 and energy > 0.5:
                for k in range(4):
                    tracks["drums"].append(dict(
                        start=bar_offset * 4 + (bars - 1) * 4 + 2 + k * 0.5,
                        dur=0.5, drum="tom", vel=0.5 + 0.12 * k))
        bar_offset += bars

    # swing / humanise
    swing = g["swing"]
    for name, evs in tracks.items():
        for e in evs:
            e["start"] = _swing(e["start"], swing, rng)
            if name not in ("drums",):
                e["start"] += float(rng.normal(0, 0.006))
                e["vel"] = float(np.clip(e["vel"] * rng.normal(1.0, 0.06), 0.1, 1.2))

    # ---- render each track to mono --------------------------------------
    def mono_track(name):
        buf = np.zeros(total_samples, dtype=np.float32)
        evs = tracks[name]
        if not evs:
            return buf
        if name == "drums":
            for e in evs:
                fn = DRUMS.get(e["drum"])
                if fn is None:
                    continue
                s = int(e["start"] * spb * sr)
                if s >= total_samples:
                    continue
                x = fn(sr, e["vel"], rng)
                _add(buf, s, x)
            return compress(buf, sr, -12.0, 2.0, 40.0)
        if name == "vocal":
            if not want_vocal:
                return buf
            notes = []
            for e in evs:
                notes.append(dict(start=e["start"] * spb, dur=max(0.08, e["dur"] * spb),
                                  freq=float(midi_to_freq(e["midi"])), vel=e["vel"]))
            notes.sort(key=lambda z: z["start"])
            x = sing_phrase(notes, sr, lyrics=lyrics or None, voice=voice, rng=rng)
            x = highpass(x, sr, 110)
            x = compress(x, sr, -18.0, 3.0, 30.0)
            _add(buf, 0, x)
            return buf
        inst = {"keys": keys_inst, "pad": pad_inst, "bass": bass_inst,
                "arp": arp_inst, "lead": lead_inst}.get(name)
        if not inst or inst not in INSTRUMENTS:
            return buf
        fn = INSTRUMENTS[inst]
        for e in evs:
            s = int(e["start"] * spb * sr)
            if s >= total_samples:
                continue
            dur = max(0.05, e["dur"] * spb)
            x = fn(float(midi_to_freq(e["midi"])), dur, sr, vel=e["vel"], rng=rng)
            _add(buf, s, x)
        return buf

    # ---- mix -------------------------------------------------------------
    dry = np.zeros((2, total_samples), dtype=np.float32)
    send = np.zeros(total_samples, dtype=np.float32)
    stems = {}
    kick_times = None
    if g.get("pump", 0) > 0:
        kick_times = [int(e["start"] * spb * sr) for e in tracks["drums"] if e["drum"] == "kick"]

    for i, name in enumerate(TRACK_ORDER):
        if progress:
            progress(0.15 + 0.6 * i / len(TRACK_ORDER), "rendering %s" % name)
        buf = mono_track(name)
        if not np.any(buf):
            continue
        gain, pan, snd = g["mix"].get(name, (0.5, 0.0, 0.3))
        if name == "vocal" and not want_vocal:
            continue
        # loudness match before the fader
        rms = float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))
        if rms > 1e-6:
            buf = buf * (TRACK_LEVEL.get(name, 0.07) / rms)
        # per-track polish
        if name == "arp":
            buf = delay(buf, sr, 0.25, 0.25, 0.18)
        if name == "lead":
            buf = chorus(buf, sr, 6.0, 0.4, 0.25)
        if name == "vocal":
            buf = delay(buf, sr, spb * 0.75, 0.22, 0.14)
        if kick_times and name in ("keys", "pad", "arp", "bass"):
            depth = g["pump"]
            env = np.ones(total_samples, dtype=np.float32)
            ln = int(0.20 * sr)
            for kt in kick_times:
                a = kt
                b = min(total_samples, kt + ln)
                if a < total_samples:
                    env[a:b] *= (1.0 - depth * np.linspace(1.0, 0.0, b - a) ** 1.5)
            buf = buf * env
        gl, gr = pan_gains(pan)
        dry[0] += buf * (gain * gl)
        dry[1] += buf * (gain * gr)
        if snd > 0:
            send += buf * (gain * snd)
        if params.get("stems"):
            stems[name] = buf.copy()

    if progress:
        progress(0.8, "mixing and reverb")
    if np.any(send):
        ir_l = reverb_ir(sr, g["reverb"], 2.9, np.random.default_rng(seed + 1))
        ir_r = reverb_ir(sr, g["reverb"], 3.1, np.random.default_rng(seed + 2))
        wet_l = convolve(send, ir_l)
        wet_r = convolve(send, ir_r)
        n = min(dry.shape[1], wet_l.size)
        dry[0, :n] += wet_l[:n] * 0.9
        dry[1, :n] += wet_r[:n] * 0.9

    mixd = highpass(dry, sr, 28.0, order=1)
    mixd = compress(mixd, sr, -14.0, 2.0, 90.0)
    mixd = soft_clip(mixd, 1.05)
    mixd = normalize(mixd, 0.94)
    mixd = fade(mixd, sr, 0.02, 1.6)
    if progress:
        progress(0.95, "mastering")

    info = dict(title=params.get("title") or theory.random_title(rng),
                genre=genre, genre_label=GENRE_LABELS[genre], bpm=bpm,
                key=theory.key_name(root, scale), root=root, scale=scale,
                seed=seed, seconds=round(mixd.shape[1] / sr, 2),
                bars=total_bars, progression=progression,
                sections=[(s["name"], s["bars"]) for s in sections],
                voice=(VOICE_LABELS.get(voice, voice) if want_vocal else "instrumental"),
                voice_key=voice, vocal=want_vocal,
                instruments=dict(keys=keys_inst, pad=pad_inst, bass=bass_inst,
                                 arp=arp_inst, lead=lead_inst),
                lyrics=lyrics)
    return mixd, info, stems
