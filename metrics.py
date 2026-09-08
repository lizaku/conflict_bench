"""Metrics with the project's confound controls built in.

Non-negotiables (from the v3 postmortems):
  1. Detection AUROC is ALWAYS GroupKFold-by-relation, and ALWAYS reported
     next to the relation-base-rate-only AUROC (0.822 vs 0.936 lesson).
  2. Steering effects are ALWAYS reported vs the method's matched control
     (norm-matched random direction for act_add; alpha=0 for decoding).
  3. Every method table carries the triplet:
     detection AUROC | causal effect vs control | prompting-baseline effect.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def grouped_auroc(scores, labels, groups, n_splits=5):
    """AUROC on held-out relation groups + base-rate control AUROC."""
    scores, labels, groups = map(np.asarray, (scores, labels, groups))
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auroc_std": float("nan"),
                "base_rate_auroc": float("nan"), "n_folds": 0,
                "note": "not enough relations or a single label class"}
    gkf = GroupKFold(n_splits=n_splits)
    aucs, base_aucs = [], []
    for tr, te in gkf.split(scores, labels, groups):
        if len(np.unique(labels[te])) < 2:
            continue
        aucs.append(roc_auc_score(labels[te], scores[te]))
        # base-rate predictor: score every item by its relation's TRAIN rate
        tr_rate = pd.Series(labels[tr]).groupby(pd.Series(groups[tr])).mean()
        base = np.array([tr_rate.get(g, tr_rate.mean()) for g in groups[te]])
        base_aucs.append(roc_auc_score(labels[te], base))
    if not aucs:
        return {"auroc": float("nan"), "auroc_std": float("nan"),
                "base_rate_auroc": float("nan"), "n_folds": 0,
                "note": "every fold was single-class"}
    return {"auroc": float(np.mean(aucs)),
            "auroc_std": float(np.std(aucs)),
            "base_rate_auroc": float(np.mean(base_aucs)),
            "auroc_margin_over_base": float(np.mean(aucs) - np.mean(base_aucs)),
            "n_folds": len(aucs)}


def steering_summary(records):
    """Per (method, target, factor): flip rate, delta-margin, vs control.

    `flip_rate` is over all items; `flip_rate_flippable` is over the items
    that could move (those not already on the target side of zero) - both are
    reported because the first is deflated by whatever the base rate happens
    to be for that relation mix.

    `flip_rate_raw` is the same verdict on the uncorrected margin.  The R
    offset is a per-item constant, so it cancels out of delta-margin entirely
    - `d_margin` is identical either way, and there is deliberately no
    `d_margin_raw` column pretending otherwise.  What the correction changes
    is which side of zero an item starts and ends on, i.e. the flip rate.  A
    large gap between `flip_rate` and `flip_rate_raw` means the flips are
    being manufactured (or hidden) by the passage template rather than by the
    intervention.
    """
    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        return df
    df["d_margin"] = df.margin_after - df.margin_before
    df["d_margin_ctrl"] = np.where(
        df.control_margin_after.notna(),
        df.control_margin_after - df.margin_before, np.nan)
    df["flippable"] = [bool((e or {}).get("flippable", True))
                       for e in df.get("extras", [{}] * len(df))]
    df["flip_flippable"] = np.where(df.flippable, df.flipped, np.nan)
    if "flipped_raw" not in df:
        df["flipped_raw"] = np.nan
    df["flipped_raw"] = df.flipped_raw.astype(float)
    out = (df.groupby(["method", "target", "factor"])
             .agg(flip_rate=("flipped", "mean"),
                  flip_rate_raw=("flipped_raw", "mean"),
                  flip_rate_flippable=("flip_flippable", "mean"),
                  n_flippable=("flippable", "sum"),
                  d_margin=("d_margin", "mean"),
                  d_margin_control=("d_margin_ctrl", "mean"),
                  margin_before=("margin_before", "mean"),
                  margin_before_raw=("margin_before_raw", "mean"),
                  r_offset=("r_offset", "mean"),
                  fluency=("fluency", "mean"),
                  n=("item_id", "count"))
             .reset_index())
    out["specific_effect"] = out.d_margin - out.d_margin_control.fillna(0)
    return out


def dose_response(summary_df, method):
    """Factor sweep for one method - the AxBench Fig-4 analogue:
    x = fluency/capability proxy, y = flip rate (Pareto path)."""
    return summary_df[summary_df.method == method].sort_values("factor")


def triplet_report(detection_results: dict, steering_df: pd.DataFrame,
                   prompt_method="prompt_instruct"):
    """The mandatory joint table. detection_results: {method: grouped_auroc dict}."""
    # one row per method: the primary (position, condition) cell of the sweep,
    # so the joint table does not multiply out into every probe site
    primary = {}
    for key, d in detection_results.items():
        base = d.get("method", key)
        if d.get("primary", True) or base not in primary:
            primary[base] = d
    detection_results = primary
    methods = list(detection_results)
    if steering_df is not None and len(steering_df):
        methods += [m for m in steering_df.method.unique() if m not in methods]
        prompt_effect = steering_df[steering_df.method == prompt_method]
        best_prompt = (prompt_effect.groupby("target").flip_rate.max().mean()
                       if len(prompt_effect) else np.nan)
    else:
        steering_df, best_prompt = pd.DataFrame(), np.nan
    rows = []
    for m in methods:
        d = detection_results.get(m, {})
        s = steering_df[steering_df.method == m] if len(steering_df) else steering_df
        rows.append({
            "method": m,
            "auroc_grouped": d.get("auroc", np.nan),
            "auroc_raw": d.get("auroc_raw", np.nan),
            "auroc_position": d.get("position"),
            "readout_auroc": d.get("readout_auroc", np.nan),
            "auroc_over_readout": d.get("auroc_over_readout", np.nan),
            "auroc_base_rate": d.get("base_rate_auroc", np.nan),
            "auroc_margin_over_base": (d.get("auroc", np.nan)
                                       - d.get("base_rate_auroc", np.nan)),
            "best_flip_rate": s.flip_rate.max() if len(s) else np.nan,
            "best_flip_rate_raw": (s.flip_rate_raw.max()
                                   if len(s) and "flip_rate_raw" in s
                                   else np.nan),
            "best_flip_rate_flippable": (s.flip_rate_flippable.max()
                                         if len(s) else np.nan),
            "best_specific_effect": (s.specific_effect.abs().max()
                                     if len(s) else np.nan),
            "prompting_baseline_flip": best_prompt,
        })
    return pd.DataFrame(rows)
