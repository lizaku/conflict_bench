"""Evaluation: predictions + instances -> tables.  Reads files only.

    python -m conflict_bench.evaluate bench/<model>/runs/<name>

Detection (AUROC)
  question-relevant conflict   (q, c, M) -> score     positives: conflict_q
  global knowledge conflict    (c, M)    -> score     positives: conflict_global
  localization                 score per sentence     positives: sentences
                               that assert b; sentence AUROC, top-1 hit, F1

  The AUROC is given for all positives against all negatives, and then
  against each negative recipe on its own: `con` vs `irr` is easy, `con` vs
  `nov` is the real test, and a pooled number hides that.

Steering, for a direction with target t on an item:

    success    = #{ d != t and x == t } / #{ d != t }
    collateral = #{ d == t and x != t } / #{ d == t }

  "x == t" means x names t and no other candidate answer (construct/match).
  Both are always printed with their denominators.

Instances with an uncertain fact on the answer path are kept out of every
table above and reported in their own `*_uncertain` tables, labelled as if
M's weak answer counted as knowledge.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conflict_bench.construct import match
from conflict_bench.construct.common import read_jsonl


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) < 2 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


# ================================================================ detection
def _detection_frame(recs, rows, uncertain):
    R = {r["instance_id"]: r for r in recs if r["has_uncertain"] == uncertain}
    sfx = "_if_known" if uncertain else ""
    data = []
    for p in rows:
        r = R.get(p["instance_id"])
        if r is None:
            continue
        key = "conflict_global" if p["task"] == "global" else "conflict_q"
        data.append({
            "method": p["method"], "task": p["task"], "score": p["score"],
            "y": r["labels"][key + sfx], "recipe": r["recipe"],
            "format": r["format"], "k": r["features"]["k"],
            "self_consistency": r["features"]["self_consistency"],
            "fold": r["fold"], "forward_passes": p.get("forward_passes"),
            "access": p.get("access"), "oracle": p.get("oracle", False),
            "chunks": p.get("chunks"),
            "chunk_y": r["labels"]["chunk_asserts_b" + sfx]})
    return pd.DataFrame(data)


def detection_table(df) -> pd.DataFrame:
    out = []
    if df.empty:
        return pd.DataFrame()
    for (method, task), g in df.groupby(["method", "task"], sort=False):
        base = {"method": method, "task": task,
                "access": g["access"].iloc[0], "oracle": bool(g["oracle"].iloc[0]),
                "forward_passes": round(float(g["forward_passes"].mean()), 1)}

        def row(sub, slice_, pos, neg):
            out.append(dict(base, slice=slice_, positives=pos, negatives=neg,
                            auroc=auroc(sub["y"], sub["score"]),
                            n_pos=int((sub["y"] == 1).sum()),
                            n_neg=int((sub["y"] == 0).sum())))
        pos, neg = g[g["y"] == 1], g[g["y"] == 0]
        row(g, "all", "all", "all")
        for r in sorted(neg["recipe"].unique()):
            row(pd.concat([pos, neg[neg["recipe"] == r]]), "all", "all", r)
        if pos["recipe"].nunique() > 1:
            for r in sorted(pos["recipe"].unique()):
                row(pd.concat([pos[pos["recipe"] == r], neg]), "all", r, "all")
        for col in ("format", "k", "self_consistency"):
            if g[col].nunique() > 1:
                for v in sorted(g[col].unique(), key=str):
                    row(g[g[col] == v], f"{col}={v}", "all", "all")
    return pd.DataFrame(out)


def _best_f1_threshold(scores, labels):
    best = (-1.0, 0.0)
    for t in np.unique(scores):
        pred = scores >= t
        tp = float((pred & (labels == 1)).sum())
        if tp == 0:
            continue
        p, r = tp / pred.sum(), tp / (labels == 1).sum()
        best = max(best, (2 * p * r / (p + r), float(t)))
    return best[1]


def localization_table(df) -> pd.DataFrame:
    """Per-sentence scores of the question-relevant task."""
    out = []
    if df.empty:
        return pd.DataFrame()
    df = df[(df["task"] == "qr") & df["chunks"].notna()]
    for method, g in df.groupby("method", sort=False):
        S, Y, F, top1 = [], [], [], []
        for _, r in g.iterrows():
            s, y = np.asarray(r["chunks"], float), np.asarray(r["chunk_y"])
            if len(s) != len(y) or not np.isfinite(s).all():
                continue
            S.append(s), Y.append(y), F.append(np.full(len(s), r["fold"]))
            if y.any():
                top1.append(bool(y[int(np.argmax(s))]))
        if not S:
            continue
        S, Y, F = map(np.concatenate, (S, Y, F))
        # F1 at a threshold chosen on the other folds, never on its own
        pred = np.zeros(len(S), bool)
        for k in np.unique(F):
            tr = F != k
            if Y[tr].any():
                pred[F == k] = S[F == k] >= _best_f1_threshold(S[tr], Y[tr])
        tp = float((pred & (Y == 1)).sum())
        prec = tp / pred.sum() if pred.sum() else 0.0
        rec = tp / (Y == 1).sum() if (Y == 1).sum() else 0.0
        out.append({"method": method, "sentence_auroc": auroc(Y, S),
                    "top1_hit": float(np.mean(top1)) if top1 else float("nan"),
                    "n_with_target": len(top1),
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    "n_sentences": int(len(S)), "n_positive": int(Y.sum())})
    return pd.DataFrame(out)


# ================================================================= steering
def _steering_frame(recs, rows, uncertain):
    R = {r["instance_id"]: r for r in recs if r["has_uncertain"] == uncertain}
    key = "conflict_q_if_known" if uncertain else "conflict_q"
    data = []
    for p in rows:
        r = R.get(p["instance_id"])
        if r is None or not r.get("d"):
            continue
        cands, t = r["answers"], p["target"]
        base = {"method": p["method"], "direction": p["direction"],
                "strength": p["strength"],
                "d_hit": match.hit(r["d"]["text"], t, cands),
                "x_hit": match.hit(p["x"], t, cands),
                "c_hit": (match.hit(p["x_control"], t, cands)
                          if p.get("x_control") is not None else np.nan),
                "forward_passes": p.get("forward_passes")}
        lp, lp0 = p.get("logp"), (r.get("confidence") or {}).get("logp_qc")
        base["margin_shift"] = np.nan
        if lp and lp0 and len(cands) > 1:
            m = lambda d: d[t] - max(v for k, v in d.items() if k != t)
            base["margin_shift"] = m(lp) - m(lp0)
        sets = [r["recipe"]] + (["conflict (all)"] if r["labels"][key] else [])
        data += [dict(base, set=s) for s in sets]
    return pd.DataFrame(data)


def steering_table(df) -> pd.DataFrame:
    out = []
    if df.empty:
        return pd.DataFrame()
    for (method, direction, strength, set_), g in df.groupby(
            ["method", "direction", "strength", "set"], sort=False):
        movable, held = g[~g["d_hit"]], g[g["d_hit"]]
        ctrl = movable["c_hit"].dropna()
        out.append({
            "method": method, "direction": direction, "strength": strength,
            "set": set_,
            "success": movable["x_hit"].mean() if len(movable) else np.nan,
            "n_d_ne_t": len(movable),
            "collateral": (~held["x_hit"]).mean() if len(held) else np.nan,
            "n_d_eq_t": len(held),
            "control_success": ctrl.astype(bool).mean() if len(ctrl) else np.nan,
            "margin_shift": g["margin_shift"].mean(),
            "forward_passes": round(float(g["forward_passes"].mean()), 1)})
    df = pd.DataFrame(out)
    order = {"conflict (all)": 0}
    return df.sort_values(
        ["method", "direction", "set", "strength"],
        key=lambda c: c.map(lambda v: order.get(v, 1)) if c.name == "set" else c,
        kind="stable").reset_index(drop=True)


# ==================================================================== driver
def _wide(det) -> pd.DataFrame:
    """The detection table, one row per (method, task), one AUROC column per
    negative recipe - the form to read; the CSV keeps the long one."""
    if det.empty:
        return det
    d = det[(det["slice"] == "all") & (det["positives"] == "all")]
    fixed = ["method", "task", "access", "oracle", "forward_passes", "n_pos"]
    rows = []
    for _, g in d.groupby(["method", "task"], sort=False):
        first = g[g["negatives"] == "all"].iloc[0]
        row = {c: first[c] for c in fixed}
        row.update({f"vs {x['negatives']}": x["auroc"] for _, x in g.iterrows()})
        rows.append(row)
    wide = pd.DataFrame(rows)
    rest = sorted(c for c in wide.columns if c not in fixed and c != "vs all")
    return wide[fixed + ["vs all"] + rest]


def _fmt(df):
    return "(no rows)" if df.empty else "```\n" + df.to_string(
        index=False, float_format=lambda v: f"{v:.3f}", na_rep="–") + "\n```"


def run(run_dir, recs=None):
    run_dir = Path(run_dir)
    if recs is None:
        ids = set(json.loads((run_dir / "manifest.json").read_text())["instance_ids"])
        recs = [r for r in read_jsonl(run_dir.parent.parent / "instances.jsonl")
                if r["instance_id"] in ids]
    det = [r for f in sorted((run_dir / "predictions").glob("*.detection.jsonl"))
           for r in read_jsonl(f)]
    ste = [r for f in sorted((run_dir / "predictions").glob("*.steering.jsonl"))
           for r in read_jsonl(f)]
    tables = {}
    for tag, unc in (("", False), ("_uncertain", True)):
        d = _detection_frame(recs, det, unc)
        tables["detection" + tag] = detection_table(d)
        tables["localization" + tag] = localization_table(d)
        tables["steering" + tag] = steering_table(_steering_frame(recs, ste, unc))
    for name, df in tables.items():
        if not df.empty:
            df.to_csv(run_dir / f"{name}.csv", index=False)

    n_unc = sum(r["has_uncertain"] for r in recs)
    main = tables["detection"]
    head = _wide(main)
    report = [
        f"# Results — {run_dir.name}", "",
        f"{len(recs)} instances ({n_unc} with an uncertain fact, reported "
        f"separately below).", "",
        "## Detection — AUROC", "",
        "One row per detector and task (`qr` = question-relevant, `global` = "
        "context only). `vs all` pools every negative; `vs <recipe>` is the "
        "same positives against that recipe alone. `oracle = True` marks the "
        "controls that were told the answers. The long form, with counts per "
        "contrast, is `detection.csv`.", "",
        _fmt(head), "",
        "### By taxonomy feature", "",
        _fmt(main[main["slice"] != "all"].drop(columns=["positives", "negatives"])
             if not main.empty else main), "",
        "## Localization", "", _fmt(tables["localization"]), "",
        "## Steering", "",
        "`success` is over the `n_d_ne_t` items whose default answer was not "
        "the target; `collateral` over the `n_d_eq_t` items where it was. "
        "`control_success` is the matched control on the same items. "
        "`margin_shift` is the change in log p(target) − log p(best other).", "",
        _fmt(tables["steering"]), "",
        "## Uncertain facts (separate)", "",
        "Labels as if M's weak answer counted as knowledge.", "",
        "### Detection", "", _fmt(_wide(tables["detection_uncertain"])), "",
        "### Steering", "", _fmt(tables["steering_uncertain"]), ""]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[evaluate] tables and report.md -> {run_dir}")
    return tables


if __name__ == "__main__":
    import conflict_bench.methods  # noqa: F401
    run(sys.argv[1])
