"""iCreate Music - local web app.

A single-file, dependency-free web UI served from 127.0.0.1. Nothing leaves
the machine: the page, the audio and the generator all live locally.
"""
import argparse
import json
import os
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from . import export, song, theory
from .synth import INSTRUMENTS
from .voice import VOICE_LABELS

JOBS = {}
JOBS_LOCK = threading.Lock()
LIBRARY = []
LIB_LOCK = threading.Lock()


def _load_library(out_dir):
    """Repopulate the session library from disk so it survives a restart."""
    found = []
    try:
        for name in sorted(os.listdir(out_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(out_dir, name), "r", encoding="utf-8") as f:
                    info = json.load(f)
                wav = name[:-5] + ".wav"
                if not os.path.exists(os.path.join(out_dir, wav)):
                    continue
                info["file"] = wav
                found.append((os.path.getmtime(os.path.join(out_dir, wav)), info))
            except Exception:
                continue
    except FileNotFoundError:
        return
    found.sort(key=lambda t: -t[0])
    with LIB_LOCK:
        LIBRARY[:] = [i for _m, i in found]


def _run_job(job_id, params, out_dir, sr):
    def progress(pct, msg):
        with JOBS_LOCK:
            JOBS[job_id].update(progress=pct, message=msg)

    try:
        progress(0.05, "composing")
        audio, info, stems = song.render(params, progress=progress)
        progress(0.9, "writing file")
        name = "%s_%s_%d" % (export.safe_name(info["title"]), info["genre"], info["seed"])
        wav = os.path.join(out_dir, name + ".wav")
        export.write_wav(wav, audio, sr)
        mp3 = export.write_mp3(os.path.join(out_dir, name + ".mp3"), wav)
        if stems:
            export.write_stems(os.path.join(out_dir, name + "_stems"), stems, sr)
        info["file"] = os.path.basename(wav)
        info["mp3"] = os.path.basename(mp3) if mp3 else None
        export.write_manifest(os.path.join(out_dir, name + ".json"), info)
        with LIB_LOCK:
            LIBRARY.insert(0, info)
        with JOBS_LOCK:
            JOBS[job_id].update(status="done", progress=1.0, message="ready",
                                result=info)
    except Exception as exc:  # noqa: BLE001 - surface the error to the UI
        import traceback
        traceback.print_exc()
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", message=str(exc))


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>iCreate Music</title>
<style>
:root{--bg:#0b0d12;--card:#141822;--card2:#1b2130;--line:#242c3d;--txt:#e8ecf4;
--dim:#8b96ad;--acc:#7c5cff;--acc2:#00d3a7;--warn:#ff6b6b}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 20% -10%,#1b1f3a 0%,#0b0d12 55%);
color:var(--txt);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:26px 28px 10px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.logo{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,var(--acc),var(--acc2));
display:grid;place-items:center;font-weight:800;font-size:20px;color:#0b0d12}
h1{font-size:20px;margin:0;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:13px;margin-top:2px}
.badge{margin-left:auto;font-size:12px;color:var(--acc2);border:1px solid #14453c;
background:#0e2a26;padding:5px 10px;border-radius:20px}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:18px;padding:14px 28px 40px;max-width:1400px}
@media(max-width:980px){main{grid-template-columns:1fr}}
@media(max-width:700px){.grid3{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}
header{padding:18px 16px 6px}main{padding:10px 16px 30px}.card{padding:15px}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:0 0 14px}
label{display:block;font-size:12px;color:var(--dim);margin:0 0 5px;letter-spacing:.02em}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
select,input[type=text],input[type=number],textarea{
width:100%;background:var(--card2);color:var(--txt);border:1px solid var(--line);
border-radius:10px;padding:9px 11px;font:inherit;font-size:14px;outline:none}
select:focus,input:focus,textarea:focus{border-color:var(--acc)}
textarea{resize:vertical;min-height:74px;font-size:13px}
input[type=range]{width:100%;accent-color:var(--acc)}
.row{display:flex;gap:10px;align-items:end}
.btn{background:linear-gradient(135deg,var(--acc),#5b3fe0);color:#fff;border:0;
border-radius:11px;padding:12px 18px;font:inherit;font-weight:600;cursor:pointer;width:100%}
.btn:hover{filter:brightness(1.08)} .btn:disabled{opacity:.5;cursor:default}
.btn.ghost{background:var(--card2);border:1px solid var(--line);color:var(--txt);font-weight:500}
.mini{padding:8px 12px;font-size:13px;width:auto}
.chip{display:inline-flex;gap:6px;align-items:center;background:var(--card2);border:1px solid var(--line);
border-radius:20px;padding:4px 11px;font-size:12px;color:var(--dim);margin:0 6px 6px 0}
.chip b{color:var(--txt);font-weight:600}
.bar{height:6px;background:var(--card2);border-radius:6px;overflow:hidden;margin-top:12px;display:none}
.bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .3s}
.msg{font-size:13px;color:var(--dim);margin-top:8px;min-height:18px}
.player{display:none;margin-top:16px}
audio{width:100%;margin-top:6px;filter:invert(0)}
.list{display:flex;flex-direction:column;gap:8px;max-height:340px;overflow:auto}
.item{background:var(--card2);border:1px solid var(--line);border-radius:11px;padding:10px 12px;cursor:pointer}
.item:hover{border-color:var(--acc)}
.item .t{font-size:13px;font-weight:600}
.item .m{font-size:11px;color:var(--dim);margin-top:2px}
.hint{font-size:12px;color:var(--dim);margin-top:10px;line-height:1.45}
.err{color:var(--warn)}
details summary{cursor:pointer;color:var(--dim);font-size:12px;margin-top:14px}
.dl{font-size:12px;color:var(--acc2);text-decoration:none;margin-right:14px}
</style></head><body>
<header>
  <div class="logo">iC</div>
  <div><h1>iCreate Music</h1><div class="sub">Offline song generator &middot; runs entirely on this machine</div></div>
  <div class="badge">100% local &middot; no internet</div>
</header>
<main>
  <div>
    <div class="card">
      <h2>Make a song</h2>
      <div class="grid3">
        <div><label>Genre</label><select id="genre"></select></div>
        <div><label>Length <span id="lenLbl" style="color:var(--txt)">90s</span></label>
          <input type="range" id="seconds" min="15" max="300" step="5" value="90"></div>
        <div><label>Tempo</label><input type="number" id="bpm" placeholder="auto" min="0" max="200"></div>
      </div>
      <div class="grid3" style="margin-top:12px">
        <div><label>Key</label><select id="key"></select></div>
        <div><label>Scale / mode</label><select id="scale"></select></div>
        <div><label>Seed</label>
          <div class="row"><input type="number" id="seed" placeholder="random" min="0">
          <button class="btn ghost mini" id="dice" title="random seed">&#127922;</button></div></div>
      </div>
      <div class="grid" style="margin-top:12px">
        <div><label>Voice</label><select id="voice"></select></div>
        <div><label>Voice octave</label><select id="voct">
          <option value="0">normal</option><option value="-1">lower</option><option value="1">higher</option></select></div>
      </div>
      <div style="margin-top:12px">
        <label>Lyrics (optional &mdash; leave empty for wordless vocals)</label>
        <textarea id="lyrics" placeholder="Write your own words here. Each word becomes a sung syllable, so short simple lines work best:&#10;city lights are calling me home"></textarea>
      </div>
      <details>
        <summary>Advanced: choose instruments</summary>
        <div class="grid3" style="margin-top:12px">
          <div><label>Keys</label><select id="keys_inst"></select></div>
          <div><label>Pad</label><select id="pad_inst"></select></div>
          <div><label>Bass</label><select id="bass_inst"></select></div>
          <div><label>Arp</label><select id="arp_inst"></select></div>
          <div><label>Lead</label><select id="lead_inst"></select></div>
          <div><label>Extras</label>
            <div style="display:flex;gap:14px;padding-top:8px;font-size:13px">
              <label style="display:flex;gap:6px;align-items:center;margin:0"><input type="checkbox" id="instrumental"> instrumental</label>
              <label style="display:flex;gap:6px;align-items:center;margin:0"><input type="checkbox" id="stems"> stems</label>
            </div></div>
        </div>
      </details>
      <div class="row" style="margin-top:16px">
        <button class="btn" id="go">Generate song</button>
        <button class="btn ghost mini" id="surprise" title="new random song">Surprise me</button>
      </div>
      <div class="bar" id="bar"><i></i></div>
      <div class="msg" id="msg"></div>
      <div class="player" id="player">
        <div id="tags"></div>
        <audio id="au" controls></audio>
        <div style="margin-top:8px"><a class="dl" id="dlwav" download>download wav</a>
          <a class="dl" id="dlmp3" download style="display:none">download mp3</a>
          <a class="dl" id="dljson" download>details json</a></div>
      </div>
      <div class="hint">Every song is built from scratch by the synth engine &mdash; no samples, no
      model downloads. The same seed always rebuilds the same song, so write it down if you like it.</div>
    </div>
  </div>
  <div>
    <div class="card">
      <h2>Session library</h2>
      <div class="list" id="lib"><div class="hint">Nothing generated yet.</div></div>
    </div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
let OPT = null, poll = null;

async function boot(){
  const r = await fetch('/api/options'); OPT = await r.json();
  const g = $('#genre');
  for (const [k,v] of Object.entries(OPT.genres)) g.add(new Option(v, k));
  const v = $('#voice');
  for (const [k,lab] of Object.entries(OPT.voices)) v.add(new Option(lab, k));
  const kk = $('#key'); kk.add(new Option('auto',''));
  OPT.keys.forEach((n,i)=>kk.add(new Option(n, i)));
  const sc = $('#scale'); sc.add(new Option('auto',''));
  OPT.scales.forEach(s=>sc.add(new Option(s.replace(/_/g,' '), s)));
  for (const id of ['keys_inst','pad_inst','bass_inst','arp_inst','lead_inst']){
    $( '#'+id ).add(new Option('genre default',''));
    OPT.instruments.forEach(n=>$( '#'+id ).add(new Option(n,n)));
  }
  g.value = 'lofi';
}
$('#seconds').oninput = e => $('#lenLbl').textContent = e.target.value + 's';
$('#dice').onclick = () => $('#seed').value = Math.floor(Math.random()*999999999);
$('#surprise').onclick = () => { $('#seed').value=''; $('#dice').click(); $('#go').click(); };

function params(){
  return {
    genre: $('#genre').value,
    seconds: +$('#seconds').value,
    bpm: +$('#bpm').value || 0,
    key: $('#key').value === '' ? -1 : +$('#key').value,
    scale: $('#scale').value,
    seed: +$('#seed').value || 0,
    voice: $('#voice').value,
    voice_octave: +$('#voct').value,
    vocal: !$('#instrumental').checked,
    stems: $('#stems').checked,
    lyrics: $('#lyrics').value,
    keys_inst: $('#keys_inst').value, pad_inst: $('#pad_inst').value,
    bass_inst: $('#bass_inst').value, arp_inst: $('#arp_inst').value,
    lead_inst: $('#lead_inst').value,
  };
}

$('#go').onclick = async () => {
  const btn = $('#go'); btn.disabled = true; btn.textContent = 'Composing...';
  $('#bar').style.display='block'; $('#bar>i').style.width='3%';
  $('#msg').textContent = 'starting'; $('#msg').className='msg';
  try{
    const r = await fetch('/api/generate', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(params())});
    const {job_id} = await r.json();
    poll = setInterval(async () => {
      const s = await (await fetch('/api/job/'+job_id)).json();
      $('#bar>i').style.width = Math.round((s.progress||0)*100)+'%';
      $('#msg').textContent = s.message || '';
      if (s.status === 'done'){ clearInterval(poll); finish(s.result); }
      if (s.status === 'error'){ clearInterval(poll); done(s.message, true); }
    }, 500);
  } catch(e){ done(e.message, true); }
};
function done(msg, err){
  const btn=$('#go'); btn.disabled=false; btn.textContent='Generate song';
  $('#bar').style.display='none';
  if (err){ $('#msg').textContent = msg; $('#msg').className='msg err'; }
}
function finish(info){
  done();
  $('#msg').textContent = '';
  $('#player').style.display='block';
  const au = $('#au'); au.src = '/audio/' + encodeURIComponent(info.file) + '?t=' + Date.now(); au.play().catch(()=>{});
  $('#tags').innerHTML = [
    ['title', info.title], ['bpm', info.bpm], ['key', info.key],
    ['voice', info.voice], ['progression', (info.progression||[]).join(' - ')]
  ].map(([k,v]) => `<span class="chip">${k} <b>${v}</b></span>`).join('');
  $('#dlwav').href = '/audio/' + encodeURIComponent(info.file);
  $('#dlwav').setAttribute('download', info.file);
  $('#dljson').href = '/audio/' + encodeURIComponent(info.file.replace(/\.wav$/, '.json'));
  $('#dljson').setAttribute('download', info.file.replace(/\.wav$/, '.json'));
  if (info.mp3){ $('#dlmp3').style.display='inline'; $('#dlmp3').href='/audio/'+encodeURIComponent(info.mp3); }
  else { $('#dlmp3').style.display='none'; }
  loadLib();
}
async function loadLib(){
  const items = await (await fetch('/api/library')).json();
  const el = $('#lib');
  if (!items.length){ el.innerHTML = '<div class="hint">Nothing generated yet.</div>'; return; }
  el.innerHTML = items.map((s,i)=>`<div class="item" onclick="play(${i})">
    <div class="t">${s.title}</div>
    <div class="m">${s.genre_label} &middot; ${s.bpm} bpm &middot; ${s.key} &middot; ${s.seconds}s &middot; seed ${s.seed}</div>
  </div>`).join('');
  el.dataset.items = JSON.stringify(items);
}
window.play = i => {
  const items = JSON.parse($('#lib').dataset.items);
  const info = items[i];
  $('#player').style.display='block';
  const au=$('#au'); au.src='/audio/'+encodeURIComponent(info.file)+'?t='+Date.now(); au.play().catch(()=>{});
  $('#tags').innerHTML = [['title',info.title],['bpm',info.bpm],['key',info.key],['voice',info.voice]]
    .map(([k,v])=>`<span class="chip">${k} <b>${v}</b></span>`).join('');
  $('#dlwav').href='/audio/'+encodeURIComponent(info.file);
  $('#dlwav').setAttribute('download', info.file);
  $('#dljson').href='/audio/'+encodeURIComponent(info.file.replace(/\.wav$/,'.json'));
  $('#dljson').setAttribute('download', info.file.replace(/\.wav$/,'.json'));
};
boot().then(async () => {
  await loadLib();
  const want = new URLSearchParams(location.search).get('song');
  if (want){
    const items = JSON.parse($('#lib').dataset.items || '[]');
    const i = items.findIndex(s => s.file === want);
    if (i >= 0) play(i);
  }
});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    out_dir = "songs"
    sr = 44100

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/options":
            return self._send(200, {
                "genres": {k: song.GENRE_LABELS[k] for k in sorted(song.GENRES)},
                "voices": VOICE_LABELS,
                "instruments": sorted(INSTRUMENTS),
                "scales": sorted(theory.SCALES),
                "keys": theory.NOTE_NAMES,
            })
        if path == "/api/library":
            with LIB_LOCK:
                return self._send(200, LIBRARY[:50])
        if path.startswith("/api/job/"):
            jid = path.split("/")[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(jid, {"status": "unknown"}))
            return self._send(200, job)
        if path.startswith("/audio/"):
            name = unquote(path[len("/audio/"):])
            safe = os.path.basename(name)
            full = os.path.join(self.out_dir, safe)
            if not os.path.isfile(full):
                return self._send(404, {"error": "not found"})
            ctype = "audio/wav"
            if safe.endswith(".mp3"):
                ctype = "audio/mpeg"
            elif safe.endswith(".json"):
                ctype = "application/json"
            with open(full, "rb") as f:
                data = f.read()
            return self._send(200, data, ctype)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/generate":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            params = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        clean = {}
        for k, v in params.items():
            if k in ("genre", "scale", "voice", "lyrics", "keys_inst", "pad_inst",
                     "bass_inst", "arp_inst", "lead_inst"):
                clean[k] = v or None
            elif k in ("seconds", "bpm", "key", "seed", "voice_octave"):
                clean[k] = int(float(v or 0))
            elif k in ("vocal", "stems"):
                clean[k] = bool(v)
        clean["sr"] = self.sr
        if clean.get("seconds", 90) > 300:
            clean["seconds"] = 300
        if clean.get("seed", 0) <= 0:
            clean["seed"] = theory.random_seed()
        jid = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[jid] = {"status": "running", "progress": 0.0, "message": "queued"}
        t = threading.Thread(target=_run_job,
                             args=(jid, clean, self.out_dir, self.sr), daemon=True)
        t.start()
        return self._send(200, {"job_id": jid})


def serve(out_dir="songs", host="127.0.0.1", port=7860, sr=44100, open_browser=True,
          quiet=False):
    os.makedirs(out_dir, exist_ok=True)
    _load_library(out_dir)
    Handler.out_dir = out_dir
    Handler.sr = sr
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d" % (host, port)
    if not quiet:
        print("iCreate Music running at %s" % url)
        print("  songs are saved in: %s" % os.path.abspath(out_dir))
        print("  press Ctrl+C to stop")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        if not quiet:
            print("\nstopped")
    finally:
        httpd.server_close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="icreate", description="iCreate Music local app")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--out", default="songs", help="folder for generated songs")
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)
    serve(a.out, a.host, a.port, a.sr, not a.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
