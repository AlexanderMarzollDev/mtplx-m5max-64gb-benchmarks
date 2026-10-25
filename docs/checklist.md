# Checklist: before you measure (or rely on) local LLM speed on a 64GB Mac M5 Max

Most of our bad numbers came from the machine, not from the model. Go through this list
before a benchmark and before you compare your numbers with ours.

## Power

- **High Power mode on.** System Settings → Battery → Energy Mode → *High Power*
  (`pmset -g | grep powermode` shows `2`). We switched to *Automatic* once in the middle of a
  run; numbers after that point are not comparable.
- **140 W charger on MagSafe.** The USB-C ports of the 16" MacBook Pro accept at most
  **100 W** (USB-PD 20 V × 5 A). Only MagSafe 3 negotiates the 28 V / 140 W profile.
  Check what you actually get:
  `ioreg -rn AppleSmartBattery | grep -o '"AdapterDetails" = {[^}]*' | grep -o '"Watts"=[0-9]*'`
  - With 100 W and the GPU at 98 %, only about **7 W** reached the battery. It sat at 7 %.
  - With 140 W it charged at ~109 W when idle, and roughly broke even under full load
    (+0.06 A). On a long run the battery still dropped from 32 % to 26 % in 25 minutes.
- **Two chargers do not add up.** macOS uses one source, the strongest. A dock delivering
  65 W on USB-C next to MagSafe does not help.
- **Battery guard.** Our harness pauses below 40 % and resumes at 50 %; a separate watchdog
  stops everything below 20 % while discharging. Do not run multi-hour benchmarks on a laptop
  without something like that.

## Thermals

- In ~85 % of all requests mtplx reported `thermal_pressure_max = heavy`.
- Long outputs lose 10-15 % speed toward the end even when everything else is fixed
  (27B marathon: 56 → 48 tok/s over 13.6K tokens, all verify steps compiled).
- The pack authors measure with fans pinned at maximum. mtplx can do that via ThermalForge
  (`mtplx max --install`, a root daemon plus a sudoers rule). Its `smart` fan mode only ramps
  fans while a request is generating and hands them back to macOS afterwards (hold capped at
  180 s, released below 75 °C). **We did not install it, so none of our numbers use fan control.**

## A quiet machine

- **Other heavy jobs distort decode speed.** A `vitest --maxWorkers=14` run of another session
  (load 21) cut 27B code decode from 60 to 51 tok/s.
- Our harness waits for `load1 < 8` before each request and repeats any request that ended
  with `load1 > 12`. Rows with higher load are excluded from all tables.
- **Large downloads count too.** `hf download` used 4.7 GB RSS plus page cache; mtplx's memory
  guard then aborted a 43K prefill (HTTP 507, `memory_refusal`).

## Memory

- **One model at a time.** Two loaded models cut prefill by ~5x (132 vs 649 tok/s measured).
- Keep at least ~30 % memory free before a long prompt (`memory_pressure`).
- Budget for the 27B: ~23 GB short prompts, ~37 GB at 43K, ~42 GB at 125K (weights + KV +
  session cache). 164K hit 57.8 GB and was refused (see [long context](../README.md#context-window)).
- Cap the session cache when you go long: `MTPLX_SESSION_BANK_MAX_BYTES=4G` or `8G`.

## Measurement hygiene

- **Warm up once** after each server start (kernel compiles, caches).
- **Defeat the prefix cache** unless you want to measure it: put a random number at the start
  of the prompt.
- **Same sampling everywhere.** Our 27B runs use the pack's sampling (temperature 1.0,
  top-p 0.95, top-k 20). Greedy decoding raises MTP acceptance and makes numbers look better.
- **Separate code and prose.** The same model and setting gives 65 tok/s on code and
  47 tok/s on German prose, because prose is harder to predict (see acceptance chart).
- **Read the server's own numbers.** mtplx writes a JSON line per request to
  `~/.mtplx/logs/request-log-<port>.jsonl` with decode/prefill speed, acceptance per draft
  position, compiled-verify calls, peak memory and thermal pressure. Our harness reads it.
- **Repeat at least twice** and look at first-256 vs last-256 token speed: a falling tail
  means thermals or a fallback path, not the model.
