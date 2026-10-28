"""iCreate Music - a fully offline song generator.

Everything is synthesised locally from maths: no cloud, no API keys, no
downloaded samples, unlimited songs from unlimited seeds.
"""
__version__ = "1.0.0"
__app_name__ = "iCreate Music"

from . import theory, synth, voice, song, export, separate, train_voice  # noqa: F401

__all__ = ["theory", "synth", "voice", "song", "export", "separate",
           "train_voice", "__version__"]
