#!/usr/bin/env python3
"""analyze.py - summarize mtpbench result files into Markdown tables.

  python3 bench/analyze.py results/2026-10-24_27b-short-context.json [more files ...]

Per (label, task): mean decode tok/s, prefill tok/s, MTP acceptance, first/last-256-token
decode, compiled-verify share, peak memory, needle hits. Failed, empty and load-disturbed
samples (load1 > 12 or error) are left out and counted separately.
"""
import json, statistics as st, sys
from collections import OrderedDict

NOISY = 12


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.mean(xs), 1) if xs else None


def main(paths):
    groups, dropped = OrderedDict(), 0
    for p in paths:
        data = json.load(open(p))
        for r in (data["rows"] if isinstance(data, dict) else data):   # exported files wrap rows
            if r.get("error") or not r.get("completion_tokens") or (r.get("load1") or 0) > NOISY:
                dropped += 1
                continue
            groups.setdefault((r["label"], r["task"]), []).append(r)
    print("| config | task | n | prompt tok | out tok | decode tok/s | prefill tok/s | accept | "
          "first/last 256 | compiled verify | peak GB | needle |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (label, task), rs in groups.items():
        cv = [r["cv_compiled"] / r["cv_calls"] for r in rs if r.get("cv_calls")]
        needle = [r["needle_ok"] for r in rs if r.get("needle_ok") is not None]
        accs = [r["accept"] for r in rs if isinstance(r.get("accept"), (int, float))]
        acc = round(st.mean(accs), 2) if accs else None
        print(f"| {label} | {task} | {len(rs)} | {mean([r.get('prompt_tokens') for r in rs]):.0f} | "
              f"{mean([r['completion_tokens'] for r in rs]):.0f} | **{mean([r['decode_tok_s'] for r in rs])}** | "
              f"{mean([r.get('prefill_tok_s') for r in rs]) or '-'} | {f'{acc:.2f}' if acc else '-'} | "
              f"{mean([r.get('first256') for r in rs])}/{mean([r.get('last256') for r in rs])} | "
              f"{f'{100 * st.mean(cv):.0f} %' if cv else '-'} | {max(r.get('peak_gb') or 0 for r in rs)} | "
              f"{f'{sum(needle)}/{len(needle)}' if needle else '-'} |")
    print(f"\n{dropped} sample(s) excluded (error, empty response or load1 > {NOISY}).")


if __name__ == "__main__":
    main(sys.argv[1:])
