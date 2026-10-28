"""Android entry point: serve the iCreate Music web UI on localhost.

Called from MainActivity.java through Chaquopy. Everything runs on the
device: no network, no server, no account.
"""


def serve_mobile(out_dir, port=8000):
    from . import webapp
    webapp.serve(out_dir=out_dir, host="127.0.0.1", port=int(port),
                 open_browser=False, quiet=True)
