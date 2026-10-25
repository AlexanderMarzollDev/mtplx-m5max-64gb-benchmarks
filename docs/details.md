# What we found inside mtplx 2.12.2 (and why the settings work)

All file references are to the installed package (`site-packages/mtplx/`), version 2.12.2,
MLX 0.32.2, macOS 27.0.1, MacBook Pro M5 Max (40-core GPU, 64 GB).

## 1. Profiles: who gets the fast kernels

- Without `--profile`, the Qwen3.8-27B packs resolve to **turbo**, everything else to
  **sustained** (`commands/public.py`, server banner in the log).
- Turbo adds verify-specialised quantized matmul kernels and the **compiled verify** step,
  but only for **4-bit and 8-bit affine** trunks. Quote from `profiles.py`:
  *"4-bit and 8-bit affine models; 6-bit models silently run the stock path."*
  So a 6-bit pack gains nothing from `--profile turbo`.

## 2. The compiled-verify growth reserve (the +10 % for the 27B)

- Compiled verify keeps a fixed-capacity graph for the KV window it verifies against. Its
  default growth reserve is **512 tokens**. Once a request has generated past it, the request
  is demoted to the eager path for the rest of its life (`growth_budget_exhausted`).
- In the request log of a 7,679-token code answer: `compiled_verify.calls = 2202`,
  `compiled_calls = 219`, `fallback_calls = 1983`. Decode fell from ~60 to ~44 tok/s over the
  answer (first-256 vs last-256 window).
- Fix (both are operator overrides the profile respects, `profiles.py`
  `PROFILE_ENV_USER_OVERRIDE_KEYS`):

  ```
  MTPLX_COMPILED_VERIFY_GROWTH_RESERVE=16384
  MTPLX_COMPILED_VERIFY_MAX_CONTEXT=262144   # default fence is 32768
  ```

  Every verify step stays compiled. Measured: code 59.3 → 64.9 tok/s, prose 44.3 → 46.7,
  43K context 41.2 → 45.4, 125K context 37.1 → 38.3-38.8. Cost: about +1 GB peak memory.
  8K reserve is as fast on short answers but runs out on 14K-token outputs (63 % compiled).
- The reserve is clamped to the request's `max_tokens` (+6), so it only costs memory when a
  long answer is actually requested.

## 3. KV-cache quantization slows the 27B down

- With `--paged-kv-quantization q8`, the dense decode layouts are off and compiled verify
  refuses the quantized paged KV.
- Measured on 22 Oct: 27B with KV q8 15.1 tok/s vs 21.3 without at ~40K tokens (pack depth);
  short prompts looked better with q8 in that early run, but the long-context loss decides it.
- The 35B-A3B needs only 20 KiB KV per token (vs 64 KiB for the 27B), so q8 saves little there.

## 4. Draft depth

- **27B (dense):** depth 3 (the pack default and maximum) is fastest everywhere, although
  depth 2 accepts more (code 0.90 vs 0.83 per drafted token). A verify step costs mostly one
  pass over ~15 GB of weights, so checking 4 positions instead of 3 is almost free.
  Tokens per verify step: code 3.41, 125K docs 3.23, 43K summary 2.98, German prose 2.42.
- **35B-A3B (MoE), 22 Oct:** depth 1 instead of the pack's default was ~30 % faster
  (short: 91-95 vs 70-77 tok/s; 40K: 68-73 vs 56-67). Every extra draft position pulls more
  expert weights into the verify pass.
- **Adaptive depth** (`--adaptive-policy expected_value`, 35B-A3B), 23 Oct, alternating A/B runs:
  short prompts 78.0 (adaptive, start depth 2) vs 65.7 tok/s (fixed depth 1), ~40K context
  53.1 vs 57.8. Mixed, and the first unpaired series disagreed; we kept fixed depth 1.

## 5. Long context on 64 GB

- **KV per token:** 27B 64 KiB (16 full-attention layers × K+V × 4 KV heads × 256 × bf16;
  the other 48 layers are Gated DeltaNet with constant state). 35B-A3B: 20 KiB.
- **Dense-decode ceiling ("147K decode cliff"):** above it mtplx repages the decode KV and the
  packed fast verify lane is excluded; the developers measured 16-18 → 12 tok/s.
  `ceiling = RAM × MTPLX_DENSE_DECODE_RAM_PERCENT / bytes_per_token`, capped by the context
  window. On 64 GB at the default 15 %: **157,286 tokens** for the 27B.
  `MTPLX_DENSE_DECODE_RAM_PERCENT=16` lifts it to 167,772 (`generation.py`, `_dense_decode_max_context`).
- **KV growth follows the window.** With a 262K window a 164K-token request grew the paged KV
  from 168,192 to 252,288 rows in one step (+15.7 GiB) and was refused at 57.8 GB peak
  (`results/2026-10-24_27b-165k-attempt.json`). Set the window close to what you need.
- **Our 27B long-context setting** (not yet re-measured end to end at 160K):
  context window 163,840, `MTPLX_DENSE_DECODE_RAM_PERCENT=16`,
  `MTPLX_SESSION_BANK_MAX_BYTES=8G`, plus the compiled-verify settings above.
- **Prefill** of the 27B at 125K: ~560 tok/s, so a cold 125K prompt takes ~3.7 min.
  `MTPLX_PREFILL_CHUNK_SIZE_DENSE/REPAGE=4096` instead of 2048 changed nothing (562 vs 561).
- **Reading still works:** a code word hidden at 60 % depth of a 125K prompt was found in
  10 of 10 runs, across every setting including AR.

## 6. Prefix cache across restarts

- mtplx keeps a RAM session bank and an SSD session cache. A 125K entry is ~8 GB. With the
  SSD cache capped at 32 GB, the reconcile pass at the next start deleted 8.6 GB and the same
  prompt missed (`cache_miss_reason = ssd_prefix_miss`). If you work with very long prompts and
  restart the server often (idle shutdown), give the SSD cache more room.

## 7. Memory guard

- mtplx refuses work rather than letting macOS swap: HTTP 507 `memory_refusal` with reasons
  such as `compressor_runaway` or `under_abort_floor`. Our harness treats empty answers as
  failures and retries after a pause.

## 8. Other engines and drafters

- **oMLX 0.7.1rc1:** with MTP enabled the 35B-A3B decoded fastest in the short micro test
  (171 tok/s), but prefill at 8K was lower than mtplx (2,139 vs 3,489 tok/s) and the four agent
  tasks took 231 s vs 208 s. Without MTP the 27B was slow (24 tok/s, 1,338 s for the tasks).
- **LM Studio (MLX engine):** ignores MTP (mlx-engine#370). 27B: 32.5 tok/s, 2 hidden tests failed.
- **lithos-metal 0.1.2** (27B NVFP4 + LithosAI DSpark draft): 77 tok/s short decode, but prefill
  only 323-386 tok/s and one hidden test failed. As of 25 Oct 0.1.2 is still the latest release.
- **DSpark / DFlash 2:** separate draft models (DSpark for Qwen3.8-27B is 1.86B params, built for
  SGLang on NVIDIA). MLX ports exist (`mlx-dspark`, `z-lab/dflash`); their published numbers are from
  an M4 Pro (27B 4-bit DFlash 2: 14.7 → 40.0 tok/s, 1.85x at 32K). We have not tested them here.
- The 27B agent-task times in the engine table (22 Oct) predate the compiled-verify fix.
