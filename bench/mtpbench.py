#!/usr/bin/env python3
"""mtpbench.py - decode/prefill/MTP benchmark matrix for mtplx on Apple Silicon.

For every config the server is started fresh, warmed up once, then every workload runs
REPS times. Besides the client-side streaming timing it reads mtplx's own request log
(~/.mtplx/logs/request-log-<port>.jsonl): decode and prefill tok/s, MTP acceptance per
depth, decode speed of the first/last 256 tokens, compiled-verify share, peak memory,
thermal pressure, prefix-cache source. Every prompt starts with a random number so no
prefix cache can be reused (except the explicit warm workload `xlw`).

  python3 bench/mtpbench.py bench/configs/27b_short.json results/27b_short.json

Config: [{"label": "...", "model": "<HF repo id or path>", "flags": ["--depth", "3"],
          "env": {"MTPLX_COMPILED_VERIFY_GROWTH_RESERVE": "16384"}, "context": 262144}]

Environment:
  REPS (2), REPS_MAP ('{"xl":1}'), WORKLOADS (code,prosa,lang,marathon,xl,xlw)
  CORPUS_DIR   folder with text/code files for the long-context workloads (lang, xl, xlw);
               any large source tree works, files are concatenated in sorted order
  XL_CHARS     corpus size for xl/xlw in characters (420000 ~ 125K tokens of code)
  QUIET_LOAD (8), NOISY_LOAD (12), MIN_FREE_PCT (30), BATT_MIN (40), BATT_RESUME (50)
  MTPLX        path of the mtplx binary (default: `mtplx` on PATH), PORT (18100)

Guards: waits for a quiet machine (load average) and free memory before each request,
repeats requests disturbed by load or answered empty (mtplx memory guard), and pauses
while the battery is low so a laptop never runs flat under sustained GPU load.
"""
import json, os, random, re, shutil, signal, subprocess, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "18100"))
API = f"http://127.0.0.1:{PORT}/v1"
RLOG = os.path.expanduser(f"~/.mtplx/logs/request-log-{PORT}.jsonl")
MTPLX = os.environ.get("MTPLX") or shutil.which("mtplx") or "mtplx"
REPS = int(os.environ.get("REPS", "2"))
REPS_MAP = json.loads(os.environ.get("REPS_MAP", "{}"))
WORKLOADS = os.environ.get("WORKLOADS", "code,prosa,lang,marathon").split(",")
QUIET = float(os.environ.get("QUIET_LOAD", "8"))
NOISY = float(os.environ.get("NOISY_LOAD", "12"))
MIN_FREE = int(os.environ.get("MIN_FREE_PCT", "30"))
BATT_MIN = int(os.environ.get("BATT_MIN", "40"))
BATT_RESUME = int(os.environ.get("BATT_RESUME", "50"))
NEEDLE = "ZK-7391-ORCA"
XL = ""

# The prompts are German on purpose: German prose shows a much lower MTP acceptance than
# code, which is one of the findings. Keep them unchanged to compare against our results.
DOC_TASK = ("Nenne in der ersten Zeile den Pruefcode, der im Material steht. Danach: Schreibe zu jedem "
            "Werkzeug im Material eine ausfuehrliche deutsche Doku (Zweck, Aufruf, Optionen, Fallen, "
            "Beispiel). Gehe der Reihe nach vor und hoere nicht auf, bevor du alle Werkzeuge behandelt hast.")
TASKS = {   # name: (max_tokens, context kind, prompt)
    "code": (8000, "", "Schreibe ein vollstaendiges Python-Modul `rechnung.py`: Dataclasses fuer Rechnung "
             "und Position, Parser fuer ein CSV-Format, Validierung (USt 7/19 %, Summen), Export als "
             "JSON, und dazu pytest-Tests fuer alle Funktionen. Nur Code, mit Docstrings."),
    "prosa": (8000, "", "Erklaere ausfuehrlich auf Deutsch (mindestens 1000 Woerter), wie spekulatives "
              "Dekodieren mit Multi-Token-Prediction funktioniert, warum es auf Apple Silicon "
              "speicherbandbreitengebunden ist und welche Rolle Akzeptanzrate und Draft-Tiefe spielen."),
    "lang": (3000, "CTX", "Fasse das Material oben strukturiert zusammen: je Werkzeug zwei Saetze, "
             "danach die fuenf wichtigsten Designregeln, die sich durch alle ziehen."),
    # >120K context with long output. xl: random number first (cold prefill). xlw: random
    # number after the material, so a second request can reuse the prefix from the cache.
    "xl": (8000, "XL", DOC_TASK),
    "xlw": (8000, "XLW", DOC_TASK),
    "marathon": (16384, "", "Schreibe ein ausfuehrliches deutsches Handbuch 'Lokale KI auf dem Mac' mit "
                 "20 nummerierten Kapiteln, jedes Kapitel mindestens 400 Woerter, mit Beispielen und "
                 "Code-Schnipseln. Hoere nicht vor Kapitel 20 auf."),
}


def corpus(chars):
    base = os.environ.get("CORPUS_DIR")
    if not base: sys.exit("CORPUS_DIR is required for the lang/xl/xlw workloads")
    parts, n = [], 0
    for root, dirs, files in os.walk(os.path.expanduser(base)):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            if os.path.getsize(p) >= 200_000: continue
            try: t = f"### {os.path.relpath(p, base)}\n" + open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError): continue
            parts.append(t); n += len(t)
            if n > chars: return "\n\n".join(parts)[:chars]
    return "\n\n".join(parts)[:chars]


# ---------- machine guards ----------

def mem_free_pct():
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
        return int(out.rsplit("percentage:", 1)[1].strip().rstrip("%"))
    except Exception:
        return 100


def battery():
    """(percent, amps); amps < 0 means discharging. (100, 0) on machines without a battery."""
    try:
        out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True, timeout=10).stdout
        pct_m = re.search(r'"CurrentCapacity" = (\d+)', out)
        amp_m = re.search(r'"InstantAmperage" = (\d+)', out)
        if not pct_m or not amp_m: return 100, 0.0
        pct, amp = int(pct_m.group(1)), int(amp_m.group(1))
        if amp >= 2**63: amp -= 2**64            # ioreg prints negative values unsigned
        return pct, amp / 1000
    except Exception:
        return 100, 0.0


def wait_ready(limit=600):
    """Pause on low battery; wait for low load and free memory (other jobs skew decode
    speed, and a long prefill trips mtplx's memory guard when memory is tight)."""
    t = time.time()
    if battery()[0] < BATT_MIN:
        print(json.dumps({"info": f"battery {battery()[0]} %, waiting for {BATT_RESUME} %"}), flush=True)
        while battery()[0] < BATT_RESUME: time.sleep(30)
    q = time.time()
    while (os.getloadavg()[0] > QUIET or mem_free_pct() < MIN_FREE) and time.time() - q < limit:
        time.sleep(10)
    return round(time.time() - t)


# ---------- server ----------

def api_up():
    try:
        with urllib.request.urlopen(f"{API}/models", timeout=3) as r:
            return bool(json.load(r).get("data"))
    except Exception:
        return False


def stop_server(proc=None):
    subprocess.run([MTPLX, "stop", "--port", str(PORT)], capture_output=True, timeout=60)
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try: proc.wait(30)
        except subprocess.TimeoutExpired: proc.kill()
    for _ in range(30):
        if not api_up(): return
        time.sleep(1)


def start_server(cfg, log):
    cmd = [MTPLX, "serve", "--model", cfg["model"], "--model-id", "bench", "--host", "127.0.0.1",
           "--port", str(PORT), "--no-auth", "--context-window", str(cfg.get("context", 131072)),
           "--ssd-session-cache", "on", "--yes", *cfg.get("flags", [])]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, **cfg.get("env", {})})
    for _ in range(300):
        if api_up(): return proc
        if proc.poll() is not None: raise RuntimeError(f"mtplx exited, see server log ({' '.join(cmd)})")
        time.sleep(2)
    raise RuntimeError("mtplx not ready after 600 s")


# ---------- one request ----------

def last_log(t0):
    try:
        with open(RLOG, "rb") as f:
            f.seek(max(0, os.path.getsize(RLOG) - 400_000))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return {}
    for z in reversed(lines):
        try: d = json.loads(z)
        except Exception: continue
        return d if d.get("logged_at_s", 0) >= t0 else {}
    return {}


def one(task, ctx, temp=None):
    waited = wait_ready()
    batt0 = battery()
    max_tokens, kind, text = TASKS[task]
    nonce = random.randint(10**8, 10**9)
    if kind == "XLW":
        msg = f"Material:\n{XL}\n\nLauf {nonce}.\n\n" + text
    else:
        mat = XL if kind == "XL" else ctx
        msg = f"Lauf {nonce}.\n\n" + (f"Material:\n{mat}\n\n" if kind else "") + text
    body = {"model": "bench", "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    if temp is not None: body["temperature"] = temp
    req = urllib.request.Request(f"{API}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = {}; out_text = []
    for raw in urllib.request.urlopen(req, timeout=3600):
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        o = json.loads(line[6:])
        if o.get("usage"): usage = o["usage"]
        for c in o.get("choices") or []:
            d = c.get("delta") or {}
            if d.get("content"): out_text.append(d["content"])
            if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                first = time.time() - t0
    dt = time.time() - t0
    time.sleep(1.5)                              # the log line is written after the stream ends
    lg = last_log(t0 - 1)
    cv = lg.get("compiled_verify") or {}
    out = usage.get("completion_tokens", 0)
    drafted, acc = lg.get("drafted_tokens") or 0, lg.get("accepted_drafts") or 0
    gb = lambda b: round((b or 0) / 1e9, 1)
    content = "".join(out_text)
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": out,
            "ttft_s": round(first or 0, 2), "total_s": round(dt, 1),
            "client_decode_tok_s": round(out / max(dt - (first or 0), .01), 1),
            "decode_tok_s": round(lg.get("decode_tok_s") or 0, 1),
            "prefill_tok_s": round(lg.get("prefill_tok_s") or 0),
            "first256": round(lg.get("sliding_decode_tok_s_first_256") or 0, 1),
            "last256": round(lg.get("sliding_decode_tok_s_last_256") or 0, 1),
            "mtp_depth": lg.get("mtp_depth"), "accept": round(acc / drafted, 3) if drafted else None,
            "accepted_by_depth": lg.get("accepted_by_depth"), "drafted_by_depth": lg.get("drafted_by_depth"),
            "cv_compiled": cv.get("compiled_calls"), "cv_calls": cv.get("calls"),
            "peak_gb": gb(lg.get("peak_memory_bytes")), "active_gb": gb(lg.get("active_memory_bytes")),
            "thermal": lg.get("thermal_pressure_max"), "temp": lg.get("effective_temperature"),
            "cached_tokens": lg.get("cached_tokens"), "cache_source": lg.get("cache_source"),
            "cache_miss_reason": lg.get("cache_miss_reason"), "error_kind": lg.get("error_kind"),
            "needle_ok": (NEEDLE in content) if kind in ("XL", "XLW") else None,
            "content_chars": len(content), "load1": round(os.getloadavg()[0], 1), "waited_s": waited,
            "batt_start": batt0[0], "batt_end": battery()[0]}


def main():
    global XL
    configs, out_path = json.load(open(sys.argv[1])), sys.argv[2]
    results = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["label"], r["task"], r["rep"]) for r in results if "error" not in r}
    ctx = corpus(140_000) if "lang" in WORKLOADS else ""
    if {"xl", "xlw"} & set(WORKLOADS):          # needle at 60 % depth for a retrieval check
        XL = corpus(int(os.environ.get("XL_CHARS", "420000")))
        cut = XL.rfind("\n", 0, int(len(XL) * 0.6))
        XL = XL[:cut] + f"\n# HINWEIS: Der Pruefcode fuer diese Sammlung lautet {NEEDLE}.\n" + XL[cut:]
    log = open(os.path.splitext(out_path)[0] + ".server.log", "a")
    for cfg in configs:
        label = cfg["label"]
        if all((label, t, r) in done for t in WORKLOADS for r in range(REPS_MAP.get(t, REPS))): continue
        stop_server(); time.sleep(8)
        t = time.time()
        try: proc = start_server(cfg, log)
        except Exception as e:
            results.append({"label": label, "task": "start", "rep": 0, "error": str(e)[:400]})
            json.dump(results, open(out_path, "w"), indent=1); continue
        boot = round(time.time() - t, 1)
        one("prosa", "", cfg.get("temp"))           # warm-up: kernel compiles, caches
        for task in WORKLOADS:
            for rep in range(REPS_MAP.get(task, REPS)):
                if (label, task, rep) in done: continue
                m: dict = {}
                for attempt in range(3):
                    try: m = one(task, ctx, cfg.get("temp"))
                    except Exception as e: m = {"error": str(e)[:300]}; break
                    m["attempt"] = attempt
                    if not m.get("completion_tokens"):
                        m["error"] = "empty response"; time.sleep(60); continue
                    if m.get("load1", 0) <= NOISY: break
                m.update(label=label, model=cfg["model"], task=task, rep=rep, boot_s=boot,
                         flags=cfg.get("flags", []), env=cfg.get("env", {}))
                results.append(m); print(json.dumps(m, ensure_ascii=False), flush=True)
                json.dump(results, open(out_path, "w"), indent=1)
        stop_server(proc)


if __name__ == "__main__":
    main()
