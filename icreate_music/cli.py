"""Command line interface.

    python -m icreate_music.cli --genre lofi --seconds 60
    python -m icreate_music.cli --count 20 --out library/ --genre edm
"""
import argparse
import os
import sys
import time

from . import export, song, theory, train_voice
from .synth import INSTRUMENTS
from .voice import VOICE_LABELS


def _progress(pct, msg):
    sys.stdout.write("\r  [%3d%%] %-24s" % (int(pct * 100), msg))
    sys.stdout.flush()


def build_parser():
    p = argparse.ArgumentParser(
        prog="icreate", description="iCreate Music - offline AI song generator")
    p.add_argument("--genre", default="pop", choices=sorted(song.GENRES),
                   help="musical style")
    p.add_argument("--seconds", type=float, default=90, help="length in seconds")
    p.add_argument("--bpm", type=int, default=0, help="tempo (0 = pick for genre)")
    p.add_argument("--key", default="", help="root note, e.g. C, F#, A")
    p.add_argument("--scale", default="", help="scale/mode, e.g. minor, dorian")
    p.add_argument("--seed", type=int, default=0, help="seed (0 = random)")
    p.add_argument("--voice", default="female",
                   help="singing voice: a preset, or a trained profile slug")
    p.add_argument("--voice-octave", type=int, default=0, help="-1 / 0 / +1")
    p.add_argument("--instrumental", action="store_true", help="no vocals")
    p.add_argument("--lyrics", default="", help="text to sing (or a file path)")
    p.add_argument("--keys-inst", default="", choices=[""] + sorted(INSTRUMENTS))
    p.add_argument("--bass-inst", default="", choices=[""] + sorted(INSTRUMENTS))
    p.add_argument("--lead-inst", default="", choices=[""] + sorted(INSTRUMENTS))
    p.add_argument("--pad-inst", default="", choices=[""] + sorted(INSTRUMENTS))
    p.add_argument("--out", default="", help="output wav path (or folder for --count)")
    p.add_argument("--count", type=int, default=1, help="number of songs to make")
    p.add_argument("--stems", action="store_true", help="also write per-track stems")
    p.add_argument("--sr", type=int, default=44100, help="sample rate")
    p.add_argument("--mp3", action="store_true", help="also convert to mp3 (needs ffmpeg)")
    p.add_argument("--list", action="store_true", help="list genres, voices, instruments")
    p.add_argument("--list-voices", action="store_true", help="list trained voice profiles")
    p.add_argument("--train", default="", metavar="SONG",
                   help="train a voice from a song file (mp3/wav), then exit")
    p.add_argument("--name", default="", help="name for the trained voice")
    p.add_argument("--library", default="songs",
                   help="folder holding trained voices and stems (default: songs)")
    p.add_argument("--quiet", action="store_true")
    return p


def _key_to_int(text):
    if not text:
        return -1
    t = text.strip().replace("b", "#").upper()
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    flats = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    t = flats.get(t, t)
    return names.index(t) if t in names else -1


def make_one(args, seed, out_path, show=True):
    try:
        train_voice.load_all_into_engine(args.library)
    except Exception:
        pass
    lyrics = args.lyrics
    if lyrics and os.path.exists(lyrics):
        with open(lyrics, "r", encoding="utf-8") as f:
            lyrics = f.read()
    params = dict(
        genre=args.genre, seconds=args.seconds, bpm=args.bpm,
        key=_key_to_int(args.key), scale=args.scale, seed=seed,
        voice=args.voice, vocal=not args.instrumental,
        voice_octave=args.voice_octave, lyrics=lyrics, sr=args.sr,
        stems=args.stems,
        keys_inst=args.keys_inst or None, bass_inst=args.bass_inst or None,
        lead_inst=args.lead_inst or None, pad_inst=args.pad_inst or None,
    )
    t0 = time.time()
    audio, info, stems = song.render(
        params, progress=None if args.quiet else _progress)
    if not args.quiet:
        sys.stdout.write("\r" + " " * 40 + "\r")
    export.write_wav(out_path, audio, args.sr)
    if args.mp3:
        export.write_mp3(os.path.splitext(out_path)[0] + ".mp3", out_path)
    if stems:
        stem_dir = os.path.splitext(out_path)[0] + "_stems"
        export.write_stems(stem_dir, stems, args.sr)
    info["file"] = os.path.basename(out_path)
    info["render_seconds"] = round(time.time() - t0, 1)
    export.write_manifest(os.path.splitext(out_path)[0] + ".json", info)
    if show:
        print("  %s  [%s]  %d bpm  %s  seed %d  %.1fs of audio in %.1fs"
              % (info["title"], info["genre_label"], info["bpm"], info["key"],
                 info["seed"], info["seconds"], info["render_seconds"]))
    return info


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.train:
        name = args.name or os.path.splitext(os.path.basename(args.train))[0]
        print("iCreate Music - training voice \"%s\" from %s" % (name, args.train))
        prof = train_voice.train_voice(
            args.train, name, base=args.library,
            progress=None if args.quiet else _progress)
        if not args.quiet:
            sys.stdout.write("\r" + " " * 40 + "\r")
        print("  register      %.1f Hz (%.1f semitones)"
              % (prof["f0_median"], prof["semitone_range"]))
        print("  vocal tract   %.2f   vibrato %.1f Hz   breath %.2f   bright %.2f"
              % (prof["formant_scale"], prof["vibrato_rate"], prof["breath"], prof["bright"]))
        print("  saved         %s" % prof["path"])
        if prof.get("preview"):
            print("  preview       %s" % prof["preview"])
        if prof.get("stems"):
            print("  stems         %s" % prof["stems"])
        print("\n  use it with:  --voice %s" % prof["slug"])
        return 0

    if args.list_voices:
        profs = train_voice.list_profiles(args.library)
        if not profs:
            print("No trained voices yet. Train one with:\n"
                  "  python -m icreate_music train <song.mp3> --name \"Artist\"")
            return 0
        for pr in profs:
            print("%-22s %-24s f0 %6.1f Hz  range %4.1f semis  created %s"
                  % (pr.get("slug", "?"), pr.get("name", "?"), pr.get("f0_median", 0),
                     pr.get("semitone_range", 0), pr.get("created", "")))
        return 0

    if args.list:
        print("Genres:")
        for k in sorted(song.GENRES):
            print("  %-11s %s" % (k, song.GENRE_LABELS[k]))
        print("\nVoices:")
        for k in sorted(VOICE_LABELS):
            print("  %-12s %s" % (k, VOICE_LABELS[k]))
        print("\nInstruments:")
        print("  " + ", ".join(sorted(INSTRUMENTS)))
        print("\nScales:")
        print("  " + ", ".join(sorted(theory.SCALES)))
        return 0

    if args.count < 1:
        args.count = 1
    seeds = []
    base = args.seed or theory.random_seed()
    for i in range(args.count):
        seeds.append(base + i)

    if args.count == 1:
        out = args.out or "icreate_%s_%s.wav" % (args.genre, base)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        print("iCreate Music - generating...")
        make_one(args, base, out)
        print("  saved -> %s" % out)
        return 0

    out_dir = args.out or "icreate_library"
    os.makedirs(out_dir, exist_ok=True)
    print("iCreate Music - generating %d songs into %s/" % (args.count, out_dir))
    for i, seed in enumerate(seeds, 1):
        name = "%02d_%s_%d.wav" % (i, args.genre, seed)
        make_one(args, seed, os.path.join(out_dir, name))
    print("done: %d songs" % args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
