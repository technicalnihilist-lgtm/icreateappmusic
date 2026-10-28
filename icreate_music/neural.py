"""Optional neural lab: a trainable audio codec built on audio-encoders-pytorch.

This is NOT the generator. audio-encoders-pytorch ships codec *architectures*
(Encoder1d / Decoder1d / AutoEncoder1d) and no pretrained weights, so there is
nothing to download and nothing that makes music on its own. What it is good
for, locally:

  * train a codec on your own generated library, then
  * store a song as a compact latent (.icm.npz) and rebuild the audio from it,
  * get a latent tensor you can later feed to a generative model.

Requires torch + audio-encoders-pytorch. The rest of the app does not.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

CFG_DEFAULT = dict(sr=22050, channels=32, multipliers=[1, 2, 4, 8],
                   factors=[4, 4, 2], num_blocks=[1, 1, 1],
                   bottleneck_channels=8, crop_seconds=2.0)


def available():
    try:
        import torch  # noqa: F401
        import audio_encoders_pytorch  # noqa: F401
        return True, "torch + audio-encoders-pytorch ready"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "failed to map segment" in msg:
            msg += "  (try: ulimit -v unlimited, then rerun)"
        return False, msg


# --------------------------------------------------------------------------
# audio io
# --------------------------------------------------------------------------
def load_wav(path, target_sr=None):
    """Read a wav with the stdlib and return mono float32."""
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        x = ints.astype(np.float32) / 8388608.0
    else:
        raise ValueError("unsupported sample width %d" % sw)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if target_sr and sr != target_sr:
        x = resample(x, sr, target_sr)
    return x.astype(np.float32), sr


def resample(x, sr_in, sr_out):
    try:
        import torch
        import torchaudio.functional as F
        t = torch.from_numpy(x).unsqueeze(0)
        return F.resample(t, sr_in, sr_out).squeeze(0).numpy().astype(np.float32)
    except Exception:
        n_out = int(round(len(x) * sr_out / sr_in))
        idx = np.linspace(0, len(x) - 1, n_out)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def list_wavs(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith((".wav", ".wave")):
                out.append(os.path.join(root, f))
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_codec(cfg=None):
    import torch.nn as nn
    from audio_encoders_pytorch import AutoEncoder1d
    cfg = dict(CFG_DEFAULT, **(cfg or {}))
    model = AutoEncoder1d(
        in_channels=1,
        channels=cfg["channels"],
        multipliers=cfg["multipliers"],
        factors=cfg["factors"],
        num_blocks=cfg["num_blocks"],
        bottleneck_channels=cfg["bottleneck_channels"],
    )
    return model, cfg


def save_codec(path, model, cfg):
    import torch
    torch.save({"state": model.state_dict(), "config": cfg}, path)
    return path


def load_codec(path):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model, cfg = build_codec(blob["config"])
    model.load_state_dict(blob["state"])
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _crop(x, n, rng):
    if len(x) <= n:
        pad = np.zeros(n, dtype=np.float32)
        pad[:len(x)] = x
        return pad
    s = int(rng.integers(0, len(x) - n))
    return x[s:s + n]


def train(data_dir, steps=200, out="codec.pt", cfg=None, lr=1e-3, log=print):
    import torch
    ok, msg = available()
    if not ok:
        raise RuntimeError("neural stack unavailable: " + msg)
    cfg = dict(CFG_DEFAULT, **(cfg or {}))
    sr = cfg["sr"]
    files = list_wavs(data_dir)
    if not files:
        raise RuntimeError("no .wav files found in %s" % data_dir)
    log("training on %d file(s) at %d Hz, %d steps" % (len(files), sr, steps))
    cache = []
    for f in files[:64]:
        try:
            x, _ = load_wav(f, sr)
            if len(x) > sr:
                cache.append(x)
        except Exception as exc:  # noqa: BLE001
            log("  skip %s (%s)" % (os.path.basename(f), exc))
    if not cache:
        raise RuntimeError("no usable audio")
    model, cfg = build_codec(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    log("  model: %.2fM parameters, %.1fx downsample" % (n_par / 1e6,
                                                         float(np.prod(cfg["factors"]))))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    n_crop = int(cfg["crop_seconds"] * sr)
    t0 = time.time()
    best = 1e9
    for step in range(1, steps + 1):
        batch = np.stack([_crop(cache[i % len(cache)], n_crop, rng)
                          for i in range(2)])
        x = torch.from_numpy(batch).unsqueeze(1)  # (B,1,T)
        y = model(x)
        m = min(x.shape[-1], y.shape[-1])
        x_t, y_t = x[..., :m], y[..., :m]
        l1 = torch.mean(torch.abs(x_t - y_t))
        win = 512
        X = torch.stft(x_t.squeeze(1), win, win // 4, window=torch.hann_window(win),
                       return_complex=True).abs()
        Y = torch.stft(y_t.squeeze(1), win, win // 4, window=torch.hann_window(win),
                       return_complex=True).abs()
        spec = torch.mean(torch.abs(X - Y)) / (torch.mean(X) + 1e-6)
        loss = l1 + spec
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0 or step == 1:
            log("  step %4d/%d  loss %.4f  (l1 %.4f, spectral %.4f)  %.0fs"
                % (step, steps, float(loss.detach()), float(l1.detach()),
                   float(spec.detach()), time.time() - t0))
        best = min(best, float(loss.detach()))
    save_codec(out, model, cfg)
    log("saved codec -> %s  (best loss %.4f)" % (out, best))
    return out


# --------------------------------------------------------------------------
# compress / expand
# --------------------------------------------------------------------------
def compress(wav_path, codec_path, out_path=None):
    import torch
    model, cfg = load_codec(codec_path)
    x, _ = load_wav(wav_path, cfg["sr"])
    down = int(np.prod(cfg["factors"]))
    n = x.shape[0]
    xp = np.pad(x, (0, (-n) % down))
    t = torch.from_numpy(xp).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        z = model.encode(t)
        y = model.decode(z)
    m = min(n, y.shape[1])
    xa, ya = x[:m], y[0, 0, :m].numpy()
    num = float(np.sum((xa - xa.mean()) * (ya - ya.mean())))
    den = float(np.sqrt(np.sum((xa - xa.mean()) ** 2) * np.sum((ya - ya.mean()) ** 2)) + 1e-9)
    corr = num / den
    out_path = out_path or (os.path.splitext(wav_path)[0] + ".icm.npz")
    np.savez_compressed(out_path, latent=z.numpy().astype(np.float16),
                        config=json.dumps(cfg), source=os.path.basename(wav_path))
    A = np.abs(np.fft.rfft(xa * np.hanning(m)))
    B = np.abs(np.fft.rfft(ya * np.hanning(m)))
    try:
        spec_corr = float(np.corrcoef(A[1:], B[1:])[0, 1])
    except Exception:
        spec_corr = 0.0
    lat = z.numel()
    ratio = (x.shape[0] * 2) / max(1, lat * 2)  # 16-bit wav vs fp16 latent
    return dict(latent_file=out_path, latent_values=int(lat),
                downsample=float(np.prod(cfg["factors"])),
                waveform_correlation=round(corr, 4),
                spectral_correlation=round(spec_corr, 4),
                approx_size_ratio=round(ratio, 2))


def expand(latent_path, out_wav):
    import torch
    from . import export
    blob = np.load(latent_path, allow_pickle=False)
    cfg = json.loads(str(blob["config"]))
    model, cfg = build_codec(cfg)
    # weights come from the sibling codec file if present, else random init
    codec_path = os.path.splitext(latent_path)[0].replace(".icm", "") + ".codec.pt"
    if os.path.exists(codec_path):
        model, _ = load_codec(codec_path)
    z = torch.from_numpy(blob["latent"].astype(np.float32))
    with torch.no_grad():
        y = model.decode(z)
    audio = y[0, 0].numpy()
    export.write_wav(out_wav, audio, cfg["sr"])
    return out_wav


def main(argv=None):
    p = argparse.ArgumentParser(prog="icreate-neural",
                                description="Train/use a neural audio codec locally")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a codec on a folder of wavs")
    t.add_argument("data")
    t.add_argument("--steps", type=int, default=200)
    t.add_argument("--out", default="codec.pt")
    c = sub.add_parser("compress", help="song.wav -> latent .icm.npz")
    c.add_argument("wav")
    c.add_argument("--codec", default="codec.pt")
    c.add_argument("--out", default="")
    e = sub.add_parser("expand", help="latent -> wav")
    e.add_argument("latent")
    e.add_argument("--out", default="")
    sub.add_parser("check", help="is the neural stack available?")
    a = p.parse_args(argv)

    if a.cmd == "check":
        ok, msg = available()
        print(("OK  " if ok else "NO  ") + msg)
        return 0 if ok else 1
    if a.cmd == "train":
        train(a.data, a.steps, a.out)
        return 0
    if a.cmd == "compress":
        print(json.dumps(compress(a.wav, a.codec, a.out or None), indent=2))
        return 0
    if a.cmd == "expand":
        out = a.out or os.path.splitext(a.latent)[0].replace(".icm", "") + "_rebuilt.wav"
        print(expand(a.latent, out))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
