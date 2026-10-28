"""Music theory helpers: scales, chords, progressions, keys, human names."""
import random

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "minor":            [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "lydian":           [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
}

# Roman numeral -> (scale degree index, quality).  Quality: maj/min/dim/aug/sus
ROMAN = {
    "I": (0, "maj"), "II": (1, "min"), "III": (2, "min"), "IV": (3, "maj"),
    "V": (4, "maj"), "VI": (5, "min"), "VII": (6, "dim"),
    "i": (0, "min"), "ii": (1, "dim"), "iii": (2, "maj"), "iv": (3, "min"),
    "v": (4, "min"), "vi": (5, "maj"), "vii": (6, "maj"),
    "bII": (1, "maj"), "bIII": (2, "maj"), "bVI": (5, "maj"), "bVII": (6, "maj"),
    "IVmaj7": (3, "maj7"), "Vmaj7": (4, "maj7"), "Imaj7": (0, "maj7"),
    "i7": (0, "min7"), "ii7": (1, "min7"), "iii7": (2, "min7"), "vi7": (5, "min7"),
    "IV7": (3, "7"), "V7": (4, "7"),
}

CHORD_QUALITIES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus":  [0, 5, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "add9": [0, 4, 7, 14],
    "6":    [0, 4, 7, 9],
    "m7b5": [0, 3, 6, 10],
}

PROGRESSIONS = {
    "major": [
        ["I", "V", "vi", "IV"], ["I", "IV", "V", "I"], ["vi", "IV", "I", "V"],
        ["I", "vi", "IV", "V"], ["IV", "I", "V", "vi"], ["I", "iii", "IV", "V"],
    ],
    "minor": [
        ["i", "VI", "III", "VII"], ["i", "iv", "v", "i"], ["i", "VII", "VI", "VII"],
        ["i", "VI", "iv", "V"], ["i", "III", "VII", "VI"], ["i", "iv", "VII", "III"],
    ],
    "modal": [
        ["i7", "IV7", "i7", "IV7"], ["i", "bVII", "IV", "i"], ["I", "bVII", "IV", "I"],
        ["i", "bVI", "bIII", "bVII"], ["ii7", "V7", "Imaj7", "Imaj7"],
    ],
}


def key_name(root, scale):
    return "%s %s" % (NOTE_NAMES[root % 12], scale.replace("_", " "))


def scale_pitches(root, scale, octave=4):
    """MIDI pitches of the scale over two octaves starting at `octave`."""
    steps = SCALES[scale]
    base = 12 * (octave + 1) + root
    return [base + s for s in steps] + [base + 12 + s for s in steps]


def chord_pitches(root, scale, roman, octave=4, spread=0):
    """MIDI pitches for a roman numeral in the given key."""
    deg, qual = ROMAN.get(roman, (0, "maj"))
    steps = SCALES[scale]
    base = 12 * (octave + 1) + root + steps[deg % len(steps)]
    if deg >= len(steps):
        base += 12
    if roman.startswith("b") and deg == 1:
        base = 12 * (octave + 1) + root + 1
    elif roman.startswith("b") and deg == 6:
        base = 12 * (octave + 1) + root + 10
    elif roman.startswith("b") and deg == 2:
        base = 12 * (octave + 1) + root + 3
    elif roman.startswith("b") and deg == 5:
        base = 12 * (octave + 1) + root + 8
    ints = CHORD_QUALITIES.get(qual, CHORD_QUALITIES["maj"])
    return [base + i + 12 * spread for i in ints]


def pick_progression(scale, rng):
    family = "major" if "major" in scale or scale in ("lydian", "mixolydian") else (
        "modal" if scale in ("dorian", "phrygian") else "minor")
    if scale in ("major_pentatonic", "minor_pentatonic", "blues"):
        family = "minor" if "minor" in scale or scale == "blues" else "major"
    return list(rng.choice(PROGRESSIONS[family]))


def nearest_scale_index(pitches, midi):
    """Index in `pitches` closest to midi note."""
    return min(range(len(pitches)), key=lambda i: abs(pitches[i] - midi))


def diatonic_step(pitches, idx, steps):
    return max(0, min(len(pitches) - 1, idx + steps))


def humanize_time(beat, rng, amount=0.012):
    return beat + rng.uniform(-amount, amount)


TITLE_A = ["Neon", "Paper", "Midnight", "Silver", "Golden", "Velvet", "Glass",
           "Crimson", "Hollow", "Slow", "Electric", "Quiet", "Northern", "Copper",
           "Salt", "Wild", "Faded", "Lantern", "Marble", "Restless"]
TITLE_B = ["Rain", "Letters", "Highway", "Ocean", "Static", "Garden", "Machine",
           "Skyline", "Hours", "River", "Signal", "Dreams", "Orbit", "Window",
           "Tide", "Embers", "Motion", "Summer", "Silence", "Avenue"]


def random_title(rng):
    style = rng.random()
    if style < 0.55:
        return "%s %s" % (rng.choice(TITLE_A), rng.choice(TITLE_B))
    if style < 0.8:
        return "The %s %s" % (rng.choice(TITLE_A), rng.choice(TITLE_B))
    return "%s" % rng.choice(TITLE_B)


def random_seed():
    return random.randint(1, 999_999_999)
