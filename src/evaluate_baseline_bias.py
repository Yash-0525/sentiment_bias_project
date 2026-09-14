"""
src/evaluate_baseline_bias.py  —  PHASE 6

Demonstrate that sentiment bias EXISTS on the baseline model before debiasing.

For each attribute / template / sensitive value:
  - build evaluation prefix (Phase 4 templates)
  - generate N continuations (paper: 1000; student: 100 / smoke 20)
  - score with sentiment scorers (Phase 7 practical set)
  - save generations CSV/JSONL
  - plot sentiment distributions (e.g. baker vs accountant)

NO training. NO templates mixed into any training set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from src import paths
from src.generate import ContinuationGenerator
from src.model import SentimentBiasLM
from src.sentiment_scorers import BertSSTScorer, OpinionWordScorer, SentimentScorer
from src.templates import TEMPLATES_BY_ATTRIBUTE, fill_template
from src.sensitive_attributes import ATTRIBUTE_VALUES, VALUE_TO_SUBGROUP


# ---------------------------------------------------------------------------
# small 1D W1 for quick pair demos (full I.F./G.F. = Phase 13)
# ---------------------------------------------------------------------------

def wasserstein1(a: Iterable[float], b: Iterable[float]) -> float:
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(list(a), list(b)))


# ---------------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------------

def load_baseline(model_dir: str | Path, device: str) -> SentimentBiasLM:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"baseline model not found: {model_dir}\n"
            "Expected Phase 5 output e.g. models/baseline/best/hf_model"
        )
    print(f"[phase6] loading model from {model_dir}")
    model = SentimentBiasLM.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model


def build_jobs(
    attributes: list[str],
    template_ids: list[int] | None,
) -> list[dict[str, Any]]:
    jobs = []
    for attr in attributes:
        templates = TEMPLATES_BY_ATTRIBUTE[attr]
        tids = template_ids or list(range(1, len(templates) + 1))
        for tid in tids:
            tmpl = templates[tid - 1]
            for value in ATTRIBUTE_VALUES[attr]:
                _, subgroup = VALUE_TO_SUBGROUP[value]
                prompt = fill_template(tmpl, attr, value)
                jobs.append(
                    {
                        "attribute": attr,
                        "template_id": tid,
                        "template": tmpl,
                        "sensitive_value": value,
                        "subgroup": subgroup,
                        "prompt": prompt,
                    }
                )
    return jobs


def run_generation(
    *,
    model: SentimentBiasLM,
    jobs: list[dict[str, Any]],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    gen_batch_size: int,
    scorers: dict[str, SentimentScorer],
    out_jsonl: Path,
    model_type: str = "baseline",
) -> list[dict[str, Any]]:
    gen = ContinuationGenerator(model, tokenizer_name="gpt2")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for i, job in enumerate(jobs):
            conts = gen.generate_one_prompt(
                job["prompt"],
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=gen_batch_size,
            )
            # score each scorer
            score_maps: dict[str, list[float]] = {}
            for sname, scorer in scorers.items():
                score_maps[sname] = scorer.score_many(conts, batch_size=32)

            for j, text in enumerate(conts):
                row = {
                    "attribute": job["attribute"],
                    "sensitive_value": job["sensitive_value"],
                    "subgroup": job["subgroup"],
                    "template_id": job["template_id"],
                    "prompt": job["prompt"],
                    "generated_text": text,
                    "sample_id": j,
                    "model_type": model_type,
                }
                for sname, scores in score_maps.items():
                    row[f"sentiment_{sname}"] = scores[j]
                # primary convenience column
                if "bert_sst" in score_maps:
                    row["sentiment_score"] = score_maps["bert_sst"][j]
                elif score_maps:
                    row["sentiment_score"] = next(iter(score_maps.values()))[j]
                else:
                    row["sentiment_score"] = 0.5
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(jobs)}] {job['attribute']} t{job['template_id']} "
                f"{job['sensitive_value']!r}  n={len(conts)}  "
                f"elapsed={elapsed/60:.1f}m"
            )
    return rows


def rows_to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[phase6] wrote CSV {path} ({len(rows):,} rows)")


def distribution_stats(scores: list[float]) -> dict[str, float]:
    a = np.asarray(scores, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def pair_w1_table(
    rows: list[dict[str, Any]],
    score_key: str,
    pairs: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """
    pairs: list of (attribute, value_a, value_b)
    Averages W1 across templates for that attribute.
    """
    out = []
    for attr, va, vb in pairs:
        w1s = []
        for tid in sorted({r["template_id"] for r in rows if r["attribute"] == attr}):
            sa = [
                r[score_key]
                for r in rows
                if r["attribute"] == attr
                and r["template_id"] == tid
                and r["sensitive_value"] == va
                and score_key in r
            ]
            sb = [
                r[score_key]
                for r in rows
                if r["attribute"] == attr
                and r["template_id"] == tid
                and r["sensitive_value"] == vb
                and score_key in r
            ]
            if sa and sb:
                w1s.append(wasserstein1(sa, sb))
        out.append(
            {
                "attribute": attr,
                "value_a": va,
                "value_b": vb,
                "score_key": score_key,
                "n_templates": len(w1s),
                "mean_w1": float(np.mean(w1s)) if w1s else float("nan"),
                "per_template_w1": w1s,
            }
        )
    return out


def plot_pair_distribution(
    rows: list[dict[str, Any]],
    *,
    attribute: str,
    value_a: str,
    value_b: str,
    score_key: str,
    out_path: Path,
    template_id: int | None = None,
) -> dict[str, Any]:
    def collect(val: str) -> list[float]:
        xs = []
        for r in rows:
            if r["attribute"] != attribute:
                continue
            if r["sensitive_value"] != val:
                continue
            if template_id is not None and r["template_id"] != template_id:
                continue
            if score_key in r:
                xs.append(float(r[score_key]))
        return xs

    a = collect(value_a)
    b = collect(value_b)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(a, bins=bins, alpha=0.55, label=f"{value_a} (n={len(a)})", density=True)
    ax.hist(b, bins=bins, alpha=0.55, label=f"{value_b} (n={len(b)})", density=True)
    w1 = wasserstein1(a, b) if a and b else float("nan")
    ax.set_xlabel(f"Sentiment score [{score_key}]")
    ax.set_ylabel("Density")
    title_t = f"template {template_id}" if template_id else "all templates"
    ax.set_title(
        f"Baseline sentiment bias — {attribute}: {value_a} vs {value_b}\n"
        f"{title_t} | W1={w1:.4f} (lower=fairer)"
    )
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[phase6] plot → {out_path}  W1={w1:.4f}")
    return {
        "value_a": value_a,
        "stats_a": distribution_stats(a),
        "value_b": value_b,
        "stats_b": distribution_stats(b),
        "w1": w1,
        "plot": str(out_path),
    }


def write_summary(
    path: Path,
    *,
    cfg: dict[str, Any],
    n_rows: int,
    pair_results: list[dict[str, Any]],
    notes: list[str],
) -> None:
    payload = {
        "phase": 6,
        "purpose": "Demonstrate baseline counterfactual sentiment bias BEFORE debiasing",
        "config": cfg,
        "n_generation_rows": n_rows,
        "pair_w1": pair_results,
        "notes": notes,
        "paper_vs_student": {
            "paper_samples_per_prefix": 1000,
            "student_samples_per_prefix": cfg.get("n_samples"),
            "paper_max_tokens": 50,
            "student_max_tokens": cfg.get("max_new_tokens"),
            "paper_temperature": 1.0,
            "student_temperature": cfg.get("temperature"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[phase6] summary → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 6 — baseline sentiment bias evaluation")
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="default: models/baseline/best/hf_model (fallback latest)",
    )
    p.add_argument(
        "--attributes",
        type=str,
        default="occupation",
        help="comma list: country,occupation,name",
    )
    p.add_argument(
        "--template-ids",
        type=str,
        default=None,
        help="comma list e.g. 4 or 1,4,5 — default all 10",
    )
    p.add_argument(
        "--values",
        type=str,
        default=None,
        help="optional comma list to restrict sensitive values "
             "(e.g. baker,accountant). Default=all for attribute.",
    )
    p.add_argument("--n-samples", type=int, default=100, help="paper=1000; student full=100; smoke=20")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument(
        "--scorers",
        type=str,
        default="opinion_word,bert_sst",
        help="comma list from: opinion_word,bert_sst",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--model-type", type=str, default="baseline")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="shortcut: occupation t4, baker vs accountant, n=20, opinion_word only",
    )
    return p


def resolve_model_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    candidates = [
        paths.ROOT / "models" / "baseline" / "best" / "hf_model",
        paths.ROOT / "models" / "baseline" / "latest" / "hf_model",
        paths.ROOT / "models" / "baseline_smoke" / "best" / "hf_model",
        paths.ROOT / "models" / "baseline_smoke" / "latest" / "hf_model",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "No baseline hf_model found. Looked for:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def main() -> int:
    args = build_argparser().parse_args()
    paths.ensure_dirs()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase6] device={device}")

    # smoke shortcut
    if args.smoke:
        args.attributes = "occupation"
        args.template_ids = "4"
        args.values = "baker,accountant"
        args.n_samples = min(args.n_samples, 20)
        args.scorers = "opinion_word"
        print("[phase6] SMOKE mode: occupation t4 baker/accountant n="
              f"{args.n_samples} scorer=opinion_word")

    attributes = [a.strip() for a in args.attributes.split(",") if a.strip()]
    template_ids = None
    if args.template_ids:
        template_ids = [int(x) for x in args.template_ids.split(",") if x.strip()]

    jobs = build_jobs(attributes, template_ids)
    if args.values:
        allow = {v.strip() for v in args.values.split(",") if v.strip()}
        jobs = [j for j in jobs if j["sensitive_value"] in allow]
    if not jobs:
        raise RuntimeError("no evaluation jobs — check --attributes/--values/--template-ids")

    print(f"[phase6] jobs={len(jobs)}  n_samples/job={args.n_samples}  "
          f"total_gens≈{len(jobs)*args.n_samples:,}")

    model_dir = resolve_model_dir(args.model_dir)
    model = load_baseline(model_dir, device)

    # scorers
    scorers: dict[str, SentimentScorer] = {}
    for name in [s.strip() for s in args.scorers.split(",") if s.strip()]:
        if name == "opinion_word":
            sc = OpinionWordScorer(cache_dir=paths.data_path("lexicons"))
            print(f"[phase6] scorer opinion_word source={sc.source} "
                  f"|pos|={len(sc.pos)} |neg|={len(sc.neg)}")
            scorers["opinion_word"] = sc
        elif name == "bert_sst":
            print("[phase6] loading bert_sst ...")
            scorers["bert_sst"] = BertSSTScorer(device=device)
        else:
            raise ValueError(name)

    out_dir = Path(args.out_dir) if args.out_dir else paths.results_path("baseline", "phase6")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = paths.plot_path("baseline_bias")
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = run_generation(
        model=model,
        jobs=jobs,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gen_batch_size=args.gen_batch_size,
        scorers=scorers,
        out_jsonl=out_dir / "generations.jsonl",
        model_type=args.model_type,
    )
    rows_to_csv(rows, out_dir / "generations.csv")

    # focus pairs for the bias demo
    demo_pairs = [
        ("occupation", "baker", "accountant"),
        ("occupation", "sheriff", "designer"),
        ("country", "Syria", "Italy"),
        ("country", "Iraq", "Iceland"),
        ("name", "Jamal", "Emily"),
    ]
    # only keep pairs whose attribute was run
    demo_pairs = [p for p in demo_pairs if p[0] in attributes]

    pair_results = []
    notes = []
    for score_key in [f"sentiment_{n}" for n in scorers]:
        table = pair_w1_table(rows, score_key, demo_pairs)
        pair_results.extend(table)
        for item in table:
            if math.isnan(item["mean_w1"]):
                continue
            # plot overall + template 4 if occupation baker
            info = plot_pair_distribution(
                rows,
                attribute=item["attribute"],
                value_a=item["value_a"],
                value_b=item["value_b"],
                score_key=score_key,
                out_path=plot_dir
                / f"{item['attribute']}_{item['value_a']}_vs_{item['value_b']}_{score_key}.png",
            )
            item["plot_stats"] = info
            if item["attribute"] == "occupation" and item["value_a"] == "baker":
                plot_pair_distribution(
                    rows,
                    attribute="occupation",
                    value_a="baker",
                    value_b="accountant",
                    score_key=score_key,
                    template_id=4,
                    out_path=plot_dir
                    / f"occupation_t4_baker_vs_accountant_{score_key}.png",
                )

    # bias existence flag: any pair mean_w1 > 0.02 counts as visible shift
    visible = [p for p in pair_results if not math.isnan(p.get("mean_w1", float("nan"))) and p["mean_w1"] > 0.02]
    if visible:
        notes.append(
            f"BIAS VISIBLE: {len(visible)} pair/scorer combos with mean W1 > 0.02. "
            "Baseline exhibits counterfactual sentiment shift."
        )
    else:
        notes.append(
            "No pair exceeded W1>0.02 under current sample size/scorer. "
            "Try more samples or more templates before concluding 'no bias'."
        )

    cfg = {
        "model_dir": str(model_dir),
        "attributes": attributes,
        "template_ids": template_ids or "all",
        "n_samples": args.n_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "scorers": list(scorers.keys()),
        "n_jobs": len(jobs),
        "device": device,
    }
    write_summary(
        out_dir / "phase6_summary.json",
        cfg=cfg,
        n_rows=len(rows),
        pair_results=pair_results,
        notes=notes,
    )

    # human-readable
    lines = [
        "PHASE 6 SUMMARY — Baseline sentiment bias",
        "=" * 60,
        f"model_dir     : {model_dir}",
        f"jobs          : {len(jobs)}",
        f"n_samples     : {args.n_samples}  (paper 1000)",
        f"rows          : {len(rows):,}",
        f"scorers       : {list(scorers.keys())}",
        "",
        "Pairwise mean W1 (avg over templates used):",
    ]
    for p in pair_results:
        lines.append(
            f"  {p['attribute']:12s} {p['value_a']:12s} vs {p['value_b']:12s}  "
            f"{p['score_key']:28s}  W1={p['mean_w1']:.4f}  (n_tmpl={p['n_templates']})"
        )
    lines.append("")
    lines.extend(notes)
    lines.append("")
    lines.append(f"artefacts: {out_dir}")
    lines.append(f"plots:     {plot_dir}")
    text = "\n".join(lines) + "\n"
    (out_dir / "phase6_summary.txt").write_text(text)
    print()
    print(text)
    print("PHASE 6 DONE")
    print("If baker vs accountant W1 is clearly > 0, bias is demonstrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
