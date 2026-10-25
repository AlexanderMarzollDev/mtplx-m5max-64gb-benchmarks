#!/usr/bin/env python3
"""charts.py - render the comparison charts in docs/img/ from results/*.json (no dependencies).

  python3 bench/charts.py

Every chart is written twice (<name>-light.svg, <name>-dark.svg) so the README can pick the
one that matches the reader's GitHub theme. Values are computed from the raw rows with the
same filter as analyze.py: rows with an error, an empty output or load1 > 12 are skipped.
"""
import json, os, statistics as st


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "docs", "img")
THEMES = {  # categorical slots 1-4 + surfaces, validated for CVD separation and contrast
    "light": {"bg": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e1",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"bg": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}
FONT = "font-family='-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif'"


def rows(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p): return []
    return [r for r in json.load(open(p))["rows"]
            if not r.get("error") and r.get("completion_tokens") and (r.get("load1") or 0) <= 12]


def mean(label, task, data, key="decode_tok_s"):
    v = [r[key] for r in data if r["label"] == label and r["task"] == task and r.get(key)]
    return round(st.mean(v), 1) if v else None


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_path(x, y, w, h):
    """Bar with a 4px rounded data end (top), square at the baseline."""
    r = min(4, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} L{x + w:.1f},{y + h:.1f} Z")


def grouped(name, title, subtitle, groups, series, values, unit, note="", pct=False):
    """Vertical grouped bars. values[s][g] (None = not measured)."""
    for theme, t in THEMES.items():
        W, H, L, R, T, B = 900, 470, 64, 24, 118, 72
        pw, ph = W - L - R, H - T - B
        vmax = max(v for s in values for v in s if v is not None)
        step = next(x for x in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500) if vmax * 1.12 / 4 <= x)
        top = step * 4 if not pct else 100
        gw = pw / len(groups)
        bw = min(34, (gw - 28) / len(series) - 2)
        o = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
             f"<rect width='{W}' height='{H}' fill='{t['bg']}'/>",
             f"<text x='{L}' y='34' font-size='19' font-weight='600' fill='{t['ink']}'>{esc(title)}</text>",
             f"<text x='{L}' y='58' font-size='13' fill='{t['ink2']}'>{esc(subtitle)}</text>"]
        lx = L  # legend in one row, always present for >= 2 series
        for i, s in enumerate(series):
            o.append(f"<rect x='{lx}' y='76' width='12' height='12' rx='3' fill='{t['series'][i]}'/>")
            o.append(f"<text x='{lx + 18}' y='87' font-size='13' fill='{t['ink']}'>{esc(s)}</text>")
            lx += 26 + 7.4 * len(s)
        for k in range(5):  # recessive grid
            v = top * k / 4; y = T + ph - ph * v / top
            o.append(f"<line x1='{L}' x2='{W - R}' y1='{y:.1f}' y2='{y:.1f}' stroke='{t['grid']}' stroke-width='1'/>")
            o.append(f"<text x='{L - 8}' y='{y + 4:.1f}' font-size='11' text-anchor='end' fill='{t['ink2']}'>{v:.0f}</text>")
        o.append(f"<text x='14' y='{T + ph / 2}' font-size='12' fill='{t['ink2']}' transform='rotate(-90 14 {T + ph / 2})' text-anchor='middle'>{esc(unit)}</text>")
        for g, gname in enumerate(groups):
            x0 = L + g * gw + (gw - len(series) * (bw + 2)) / 2
            for s in range(len(series)):
                v = values[s][g]
                if v is None: continue
                h = ph * v / top; x = x0 + s * (bw + 2); y = T + ph - h
                o.append(f"<path d='{bar_path(x, y, bw, h)}' fill='{t['series'][s]}'><title>{esc(series[s])}, {esc(gname)}: {v} {esc(unit)}</title></path>")
                o.append(f"<text x='{x + bw / 2:.1f}' y='{y - 5:.1f}' font-size='11' text-anchor='middle' fill='{t['ink']}'>{v:g}</text>")
            o.append(f"<text x='{L + g * gw + gw / 2:.1f}' y='{T + ph + 22}' font-size='13' text-anchor='middle' fill='{t['ink']}'>{esc(gname)}</text>")
        o.append(f"<line x1='{L}' x2='{W - R}' y1='{T + ph}' y2='{T + ph}' stroke='{t['ink2']}' stroke-width='1'/>")
        if note:
            o.append(f"<text x='{L}' y='{H - 16}' font-size='11' fill='{t['ink2']}'>{esc(note)}</text>")
        o.append("</svg>")
        open(os.path.join(OUT, f"{name}-{theme}.svg"), "w").write("\n".join(o))


def hbars(name, title, subtitle, labels, values, unit, color=0, note="", fmt="{:g}"):
    """Horizontal single-series bars (long engine names). One series: the title names it, no legend."""
    for theme, t in THEMES.items():
        W, L, R, T, rowh = 900, 300, 70, 84, 30
        H = T + rowh * len(labels) + 46
        pw = W - L - R
        vmax = max(v for v in values if v is not None)
        o = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
             f"<rect width='{W}' height='{H}' fill='{t['bg']}'/>",
             f"<text x='24' y='34' font-size='19' font-weight='600' fill='{t['ink']}'>{esc(title)}</text>",
             f"<text x='24' y='58' font-size='13' fill='{t['ink2']}'>{esc(subtitle)}</text>"]
        for i, (lab, v) in enumerate(zip(labels, values)):
            y = T + i * rowh
            o.append(f"<text x='{L - 10}' y='{y + 18}' font-size='12.5' text-anchor='end' fill='{t['ink']}'>{esc(lab)}</text>")
            if v is None:
                o.append(f"<text x='{L + 6}' y='{y + 18}' font-size='12' fill='{t['ink2']}'>not measured</text>"); continue
            w = pw * v / vmax; r = min(4, w / 2)
            d = (f"M{L},{y + 5} L{L + w - r:.1f},{y + 5} Q{L + w:.1f},{y + 5} {L + w:.1f},{y + 5 + r} "
                 f"L{L + w:.1f},{y + 23 - r} Q{L + w:.1f},{y + 23} {L + w - r:.1f},{y + 23} L{L},{y + 23} Z")
            o.append(f"<path d='{d}' fill='{t['series'][color]}'><title>{esc(lab)}: {fmt.format(v)} {esc(unit)}</title></path>")
            o.append(f"<text x='{L + w + 6:.1f}' y='{y + 18}' font-size='12' fill='{t['ink']}'>{fmt.format(v)}</text>")
        o.append(f"<line x1='{L}' x2='{L}' y1='{T}' y2='{T + rowh * len(labels)}' stroke='{t['ink2']}' stroke-width='1'/>")
        o.append(f"<text x='{L}' y='{H - 14}' font-size='11' fill='{t['ink2']}'>{esc(unit)}{'  ·  ' + esc(note) if note else ''}</text>")
        o.append("</svg>")
        open(os.path.join(OUT, f"{name}-{theme}.svg"), "w").write("\n".join(o))


def main():
    os.makedirs(OUT, exist_ok=True)
    short, xl = rows("2026-10-24_27b-short-context.json"), rows("2026-10-24_27b-125k-context.json")
    # 1. 27B: what each setting does, per workload
    groups = ["Code", "German prose", "43K context", "125K context"]
    def col(lbl, xl_lbl):
        return [mean(lbl, "code", short), mean(lbl, "prosa", short), mean(lbl, "lang", short),
                mean(xl_lbl, "xl", xl + []) if xl_lbl else None]
    vals = [col("27b-ar", "xl-ar"), col("27b-d2", "xl-cv-d2"), col("27b-base-d3", "xl-d3"), col("27b-gr16k", "xl-cv262k")]
    grouped("27b-settings", "Qwen3.8-27B 4-bit on a 64GB Mac M5 Max: decode speed by setting",
            "mtplx 2.12.2, High Power mode, temperature 1.0; mean of clean runs",
            groups, ["No MTP (AR)", "MTP depth 2", "MTP depth 3 (pack default)", "Depth 3 + 16K compiled-verify reserve"],
            vals, "tokens per second", note="125K: depth-2 bar measured with the 16K reserve, AR and default without")
    # 2. 27B: MTP acceptance per draft position (share of verify steps that kept position n)
    def acc(lbl, task, data):
        rs = [r for r in data if r["label"] == lbl and r["task"] == task and r.get("accepted_by_depth") and r.get("cv_calls")]
        if not rs: return [None, None, None]
        return [round(100 * st.mean(r["accepted_by_depth"][i] / r["cv_calls"] for r in rs)) for i in range(3)]
    a = [acc("27b-gr16k", "code", short), acc("xl-cv262k", "xl", xl), acc("27b-gr16k", "lang", short), acc("27b-gr16k", "prosa", short)]
    grouped("27b-acceptance", "MTP draft acceptance by position, Qwen3.8-27B on a 64GB Mac M5 Max",
            "Share of verify steps in which draft token n was still accepted (depth 3)",
            ["Position 1", "Position 2", "Position 3"], ["Code", "125K-context docs", "43K-context summary", "German prose"],
            a, "percent of steps")
    # 2b. 27B: decode and peak memory by context length (prose-like long outputs at every length)
    grouped("27b-context-decode", "Qwen3.8-27B decode speed vs. context length, 64GB Mac M5 Max",
            "Long prose-like answers; tuned = MTP depth 3 + 16K compiled-verify reserve",
            ["~90-token prompt", "43K-token prompt", "125K-token prompt"],
            ["Tuned MTP", "No MTP (AR)"],
            [[mean("27b-gr16k", "prosa", short), mean("27b-gr16k", "lang", short), mean("xl-cv262k", "xl", xl)],
             [mean("27b-ar", "prosa", short), mean("27b-ar", "lang", short), mean("xl-ar", "xl", xl)]], "tokens per second")
    def peak(lbls, task, data):
        v = [r["peak_gb"] for r in data if r["label"] in lbls and r["task"] == task and r.get("peak_gb")]
        return max(v) if v else None
    hbars("27b-context-memory", "Qwen3.8-27B peak unified memory vs. context length, 64GB Mac M5 Max",
          "Includes model weights, KV cache and mtplx session cache; 165K run aborted by the memory guard",
          ["~90-token prompt", "43K-token prompt", "125K-token prompt", "164K-token prompt (aborted)"],
          [peak(("27b-gr16k",), "code", short), peak(("27b-gr16k",), "lang", short),
           peak(("xl-cv262k", "xl-d3"), "xl", xl), 57.8], "GB", color=3, note="45 GB = our budget so a browser still fits")
    # 3.-5. engines (22.10.)
    eng = json.load(open(os.path.join(RES, "2026-10-22_engine-comparison.json")))["engines"]
    eng = [e for e in eng if e["id"] != "omlx_coder"]
    SHORT = {"mtplx_q36_6": "mtplx · 35B-A3B Balance 6-bit", "mtplx_q36_8": "mtplx · 35B-A3B 8-bit",
             "mtplx_q38_4": "mtplx · 27B Speed 4-bit", "mtplx_q38_8": "mtplx · 27B Quality 8-bit",
             "omlx_q36_6": "oMLX · 35B-A3B 6-bit (no MTP)", "omlx_mtp": "oMLX + MTP · 35B-A3B 6-bit",
             "omlx_q38_4": "oMLX · 27B 4-bit (no MTP)", "lms_q38_4": "LM Studio · 27B 4-bit (no MTP)",
             "lithos": "Lithos 0.1.2 · 27B NVFP4 + DSpark"}
    lab = [SHORT.get(e["id"], e["id"]) for e in eng]
    hbars("engines-decode", "Short-prompt decode speed by engine, 64GB Mac M5 Max",
          "Single generation, 22 Oct 2026", lab, [e["decode_tok_s_short"] for e in eng], "tokens per second")
    hbars("engines-prefill", "Prefill speed at ~28K tokens by engine, 64GB Mac M5 Max",
          "Prompt processing, 22 Oct 2026 (8K value where 32K was not recorded)", lab,
          [e["prefill_tok_s_32k"] or e["prefill_tok_s_8k"] for e in eng], "tokens per second", color=1)
    hbars("engines-agent", "Time for 4 opencode agent tasks by engine, 64GB Mac M5 Max (lower is better)",
          "Bugfix, feature, refactor, explore; hidden tests passed in the label",
          [f"{l} ({e['agent_pass']})" for l, e in zip(lab, eng)], [e["agent_seconds_total"] for e in eng], "seconds", color=2)
    # 6. MoE (25.10.), only if measured
    moe = rows("2026-10-25_moe-35b-a3b.json")
    if moe:
        cfgs = [("moe-bal-q8-d1", "Balance 6-bit, depth 1, KV q8"), ("moe-s4-base", "Speed 4-bit, sustained default"),
                ("moe-s4-turbo-d1", "Speed 4-bit, turbo depth 1"), ("moe-s4-turbo-d2", "Speed 4-bit, turbo depth 2")]
        cfgs = [c for c in cfgs if any(r["label"] == c[0] for r in moe)]
        grouped("moe-packs", "Qwen3.6-35B-A3B (MoE) packs on a 64GB Mac M5 Max: decode speed",
                "mtplx 2.12.2, High Power mode; mean of clean runs", ["Code", "German prose", "43K context"],
                [c[1] for c in cfgs], [[mean(c[0], t, moe) for t in ("code", "prosa", "lang")] for c in cfgs], "tokens per second")
    # 7. Ternary Bonsai 2 27B vs. the regular 27B (25.10.)
    bon = rows("2026-10-25_bonsai2-27b.json")
    if bon:
        grouped("bonsai-vs-27b", "Ternary Bonsai 2 27B vs. Qwen3.8-27B 4-bit on a 64GB Mac M5 Max",
                "Decode speed, best setting each (27B: depth 3 + reserve; Bonsai: depth 1 + reserve)",
                ["Code", "German prose", "43K context", "125K context"], ["Qwen3.8-27B Speed 4-bit", "Ternary Bonsai 2 27B"],
                [[mean("27b-gr16k", "code", short), mean("27b-gr16k", "prosa", short), mean("27b-gr16k", "lang", short), mean("xl-cv262k", "xl", xl)],
                 [mean("bonsai-d1", "code", bon), mean("bonsai-d1", "prosa", bon), mean("bonsai-d1", "lang", bon), mean("bonsai-xl-d1", "xl", bon)]],
                "tokens per second")
        def pk(lbl, task, data):
            v = [r["peak_gb"] for r in data if r["label"] == lbl and r["task"] == task and r.get("peak_gb")]
            return max(v) if v else None
        grouped("bonsai-memory", "Peak memory: Ternary Bonsai 2 27B vs. Qwen3.8-27B, 64GB Mac M5 Max",
                "Weights + KV cache + session cache; the regular 27B was refused at 164K",
                ["~90-token prompt", "43K", "125K", "164K"], ["Qwen3.8-27B Speed 4-bit", "Ternary Bonsai 2 27B"],
                [[pk("27b-gr16k", "code", short), pk("27b-gr16k", "lang", short), pk("xl-cv262k", "xl", xl), 57.8],
                 [pk("bonsai-d1", "code", bon), pk("bonsai-d1", "lang", bon), pk("bonsai-xl-d1", "xl", bon), pk("bonsai-xxl-d1", "xl", bon)]],
                "GB", note="164K, regular 27B: 57.8 GB at the moment the memory guard refused the request")
    print("charts written to", OUT)


if __name__ == "__main__":
    main()
