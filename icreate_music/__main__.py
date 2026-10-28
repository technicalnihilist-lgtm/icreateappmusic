"""Entry point:  python -m icreate_music

Launches the local web app. Use "python -m icreate_music cli ..." for the
command line, or "python -m icreate_music.neural ..." for the neural lab.
"""
import sys

from . import cli, webapp


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("cli", "generate", "gen"):
        return cli.main(argv[1:])
    if argv and argv[0] == "train":
        return cli.main(["--train"] + argv[1:])
    return webapp.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
