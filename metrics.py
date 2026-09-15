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


_EMPTY_AUROC = {"auroc": float("nan"), "auroc_std": float("nan"),
                "base_rate_auroc": float("nan"), "accuracy": float("nan"),
                "accuracy_std": float("nan"),
                "majority_accuracy": float("nan"), "n_folds": 0}


def _best_threshold(scores, labels):
    """(threshold, accuracy) maximising accuracy of `scores > t` on THIS data.

    Only ever called on a training fold - see grouped_auroc.  A detector's
    score convention is HIGHER = more context-following, so the rule is
    always `> t`; a detector that needed the opposite sign would have been
    negated at the source (see the score convention in CLAUDE.md).
    """
    order = np.argsort(scores, kind="mergesort")
    s, y = np.asarray(scores)[order], np.asarray(labels)[order]
    n, pos = len(y), int(y.sum())
    # split i: predict 0 on s[:i], 1 on s[i:]
    correct_neg = np.concatenate([[0], np.cumsum(1 - y)])
    correct_pos = np.concatenate([[pos], pos - np.cumsum(y)])
    acc = (correct_neg + correct_pos) / n
    i = int(np.argmax(acc))
    if i == 0:
        thr = float(s[0]) - 1e-9
    elif i == n:
        thr = float(s[-1]) + 1e-9
    else:
        thr = float(0.5 * (s[i - 1] + s[i]))
    return thr, float(acc[i])


def grouped_auroc(scores, labels, groups, n_splits=5):
    """AUROC on held-out relation groups + base-rate control AUROC."""
    scores, labels, groups = map(np.asarray, (scores, labels, groups))
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(labels)) < 2:
        return dict(_EMPTY_AUROC,
                    note="not enough relations or a single label class")
    gkf = GroupKFold(n_splits=n_splits)
    aucs, base_aucs, accs, maj_accs = [], [], [], []
    for tr, te in gkf.split(scores, labels, groups):
        if len(np.unique(labels[te])) < 2:
            continue
        aucs.append(roc_auc_score(labels[te], scores[te]))
        # accuracy: threshold chosen on the TRAIN fold, applied to the test
        # fold - never the test-optimal threshold, which is a leak
        thr, _ = _best_threshold(scores[tr], labels[tr])
        accs.append(float(np.mean((scores[te] > thr).astype(int) == labels[te])))
        # the majority-class predictor from the same train fold: accuracy is
        # meaningless without it once the classes are imbalanced
        maj = int(np.mean(labels[tr]) >= 0.5)
        maj_accs.append(float(np.mean(labels[te] == maj)))
        # base-rate predictor: score every item by its relation's TRAIN rate
        tr_rate = pd.Series(labels[tr]).groupby(pd.Series(groups[tr])).mean()
        base = np.array([tr_rate.get(g, tr_rate.mean()) for g in groups[te]])
        base_aucs.append(roc_auc_score(labels[te], base))
    if not aucs:
        return dict(_EMPTY_AUROC, note="every fold was single-class")
    return {"auroc": float(np.mean(aucs)),
            "auroc_std": float(np.std(aucs)),
            "base_rate_auroc": float(np.mean(base_aucs)),
            "auroc_margin_over_base": float(np.mean(aucs) - np.mean(base_aucs)),
            "accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "majority_accuracy": float(np.mean(maj_accs)),
            "accuracy_over_majority": float(np.mean(accs) - np.mean(maj_accs)),
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


# --------------------------------------------------------------- simple view
#: the columns of simple_report, in order. Detection rows fill the detection
#: block and blank the steering one, and vice versa - one row per method, one
#: metric family per axis, so methods are read down a single column.
SIMPLE_COLUMNS = ["method", "axis", "auroc", "auroc_raw", "auroc_base",
                  "auroc_over_readout", "accuracy", "acc_majority",
                  "flip_rate", "flip_rate_raw", "flip_flippable",
                  "specific_effect", "factor", "target", "n"]


def simple_report(detection_results, steering_df=None):
    """One flat table: AUROC/accuracy for every detector, flip rate for every
    steerer.  The pilot's final table.

    Uniformity is the point: every detector is scored by the same numbers on
    the same held-out items, and every steerer by the same flip rate at its
    own best factor, so a column can be read straight down.

    The controls are columns or rows, not omissions: `auroc_base` (relation
    base rate), `auroc_over_readout` (vs logit_lens at the same site) and
    `acc_majority` sit beside every detector; `auroc_raw` / `flip_rate_raw`
    beside the R-corrected numbers; `bow`, `margin` and `logit_lens` are
    their own rows.  A detector that does not beat them has not been shown
    to work.
    """
    rows = []

    # detection: keep the primary (position, condition) cell per method, so
    # the table does not multiply out into every probe site
    primary = {}
    for key, d in (detection_results or {}).items():
        base = d.get("method", key)
        if d.get("primary", True) or base not in primary:
            primary[base] = d
    for m, d in primary.items():
        rows.append({"method": m, "axis": "detection",
                     "auroc": d.get("auroc", np.nan),
                     "auroc_raw": d.get("auroc_raw", np.nan),
                     "auroc_base": d.get("base_rate_auroc", np.nan),
                     "auroc_over_readout": d.get("auroc_over_readout", np.nan),
                     "accuracy": d.get("accuracy", np.nan),
                     "acc_majority": d.get("majority_accuracy", np.nan),
                     "n": d.get("n_items")})

    # steering: each method at the factor/target where it moved the most items
    if steering_df is not None and len(steering_df):
        rank = ("flip_rate_flippable" if "flip_rate_flippable" in steering_df
                else "flip_rate")
        for m, s in steering_df.groupby("method"):
            s = s[s[rank].notna()]
            if not len(s):
                continue
            best = s.loc[s[rank].idxmax()]
            rows.append({"method": m, "axis": "steering",
                         "flip_rate": best.get("flip_rate", np.nan),
                         "flip_rate_raw": best.get("flip_rate_raw", np.nan),
                         "flip_flippable": best.get("flip_rate_flippable",
                                                    np.nan),
                         "specific_effect": best.get("specific_effect", np.nan),
                         "factor": best.get("factor"),
                         "target": best.get("target"),
                         "n": best.get("n")})

    df = pd.DataFrame(rows, columns=SIMPLE_COLUMNS)
    if df.empty:
        return df
    order = {"detection": 0, "steering": 1}
    df["_axis"] = df.axis.map(order)
    df["_rank"] = np.where(df.axis == "detection",
                           df.auroc.fillna(-1), df.flip_flippable.fillna(-1))
    df = (df.sort_values(["_axis", "_rank"], ascending=[True, False])
            .drop(columns=["_axis", "_rank"]).reset_index(drop=True))
    num = [c for c in SIMPLE_COLUMNS
           if c not in ("method", "axis", "target", "n")]
    df[num] = df[num].astype(float).round(3)
    return df
