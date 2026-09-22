"""Metrics with the project's confound controls built in.

Non-negotiables (from the v3 postmortems):
  1. Detection AUROC is ALWAYS GroupKFold-by-relation, and ALWAYS reported
     next to the relation-base-rate-only AUROC (0.822 vs 0.936 lesson).
  2. Steering effects are ALWAYS reported vs the method's matched control
     (norm-matched random direction for act_add, a placebo instruction for
     prompt_instruct, the unsteered distribution for the decoding family).
     A method with no control gets NaN, never a silent zero.
  3. Every method table carries the triplet:
     detection AUROC | causal effect vs control | prompting-baseline effect.
  4. Every flip rate is reported with the `n_flippable` it was computed over.
     A rate over 6 items and a rate over 394 are not the same measurement,
     and collapsing a method to its best cell by argmax across targets is how
     4-of-6 came to outrank 390-of-394 in the pilot table.  Both targets are
     reported, always, as their own rows.
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
    """Per (method, target, factor, condition): flip rates, delta-margin,
    effect vs the matched control, and the S-vs-C specificity pairing.

    Every rate carries the count it was computed over.  `flip_rate` is over
    all items; `flip_rate_flippable` is over the items that could move (those
    not already on the target side of zero) and `n_flippable` says how many
    that was.  Reading a flippable rate without its n is how the pilot table
    ranked 4-of-6 items above 390-of-394.

    Three views of the flip verdict are kept: `flip_rate` on whichever margin
    the run made primary, `flip_rate_raw` on the observed margin, and
    `flip_rate_corrected` on the R-corrected one when R was scored.  The R
    offset is a per-item constant, so it cancels out of delta-margin entirely
    - `d_margin` is identical either way and there is deliberately no
    `d_margin_raw` column pretending otherwise.  What the correction changes
    is which side of zero an item starts and ends on.  A large gap between
    `flip_rate_raw` and `flip_rate_corrected` means the flips are being
    manufactured (or hidden) by the reference passage rather than by the
    intervention - which is exactly what the pilot postmortem found.

    `specific_effect` is delta-margin minus the method's matched control, and
    it is NaN when the method declared no control rather than being silently
    equated with the raw delta.

    `effect_toward_target` re-signs it so that HIGHER is always better for
    whichever target the row is about, which is what makes a use_context row
    and a use_parametric row comparable in one column.

    `conflict_specific_effect` is the S-vs-C pairing: the effect under the
    conflicting passage minus the effect under the supporting one, for the
    same method, target and dose. It answers the benchmark's steering
    question - can the intervention move the model when there IS a conflict,
    over and above what it does when there is not - and it is NaN unless the
    run scored both conditions.
    """
    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        return df
    df["condition"] = [getattr(c, "value", c) for c in df.condition]
    df["d_margin"] = df.margin_after - df.margin_before
    df["d_margin_ctrl"] = np.where(
        df.control_margin_after.notna(),
        df.control_margin_after - df.margin_before, np.nan)
    df["flippable"] = [bool((e or {}).get("flippable", True))
                       for e in df.get("extras", [{}] * len(df))]
    df["flip_flippable"] = np.where(df.flippable, df.flipped, np.nan)
    for col in ("flipped_raw", "flipped_corrected"):
        if col not in df:
            df[col] = np.nan
        df[col] = df[col].astype(float)

    out = (df.groupby(["method", "target", "factor", "condition"])
             .agg(flip_rate=("flipped", "mean"),
                  flip_rate_raw=("flipped_raw", "mean"),
                  flip_rate_corrected=("flipped_corrected", "mean"),
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
    # NaN, not zero: a missing control is an unmeasured control
    out["specific_effect"] = out.d_margin - out.d_margin_control
    out["has_control"] = out.d_margin_control.notna()
    sign = np.where(out.target == "use_context", -1.0, 1.0)
    out["effect_toward_target"] = out.specific_effect * sign
    out["d_margin_toward_target"] = out.d_margin * sign
    out["flippable_share"] = out.n_flippable / out.n.replace(0, np.nan)
    return _pair_conditions(out)


def _pair_conditions(out, conflict="C", support="S"):
    """Attach the S-vs-C specificity pairing to every conflict-condition row.

    The supporting passage is the matched control the benchmark's question
    asks for - same item, same intervention, same dose, nothing to arbitrate.
    A method whose effect survives the subtraction is resolving a conflict; a
    method whose effect vanishes is just amplifying whatever the passage said.
    """
    if "condition" not in out or support not in set(out.condition):
        out["support_d_margin"] = np.nan
        out["conflict_specific_effect"] = np.nan
        return out
    keys = ["method", "target", "factor"]
    sup = (out[out.condition == support][keys + ["d_margin", "specific_effect"]]
           .rename(columns={"d_margin": "support_d_margin",
                            "specific_effect": "support_specific_effect"}))
    out = out.merge(sup, on=keys, how="left")
    is_conf = out.condition == conflict
    sign = np.where(out.target == "use_context", -1.0, 1.0)
    out["conflict_specific_effect"] = np.where(
        is_conf, (out.d_margin - out.support_d_margin) * sign, np.nan)
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
        # the prompting baseline is read on the conflict condition only - the
        # S arm is a control, not a result
        if "condition" in steering_df:
            steering_df = steering_df[steering_df.condition == "C"]
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
#: block and blank the steering one, and vice versa - one metric family per
#: axis, so methods are read down a single column.
SIMPLE_COLUMNS = ["method", "axis", "target",
                  # detection block
                  "auroc", "auroc_base", "auroc_over_readout",
                  "auroc_raw", "auroc_corrected", "accuracy", "acc_majority",
                  # steering block
                  "effect", "conflict_specific", "flip_rate",
                  "flip_flippable", "n_flippable", "flip_rate_raw",
                  "flip_rate_corrected", "has_control", "factor",
                  # shared
                  "n", "note"]

#: rows that are controls rather than results, and what they control for
CONTROL_NOTES = {
    "bow": "control: text-only null (near-oracle under the conflict task)",
    "logit_lens": "control: readout at the probes' layer/position",
    "margin": "floor: the DV itself",
    "prompt_instruct": "baseline: prompting",
}


def simple_report(detection_results, steering_df=None, conflict_condition="C"):
    """One flat table: the detection axis by AUROC, the steering axis by
    effect and flip rate - with BOTH targets, always, as their own rows.

    Three things this table refuses to do, each of them a lesson from the
    pilot postmortem:

      - It does not collapse a steerer to its best cell across targets.  That
        argmax is how `use_context` at 4-of-6-flippable-items came to outrank
        `use_parametric` at 390-of-394, and how a method whose context arm was
        a no-op by construction still got a plausible-looking row.  Every
        (method, target) pair is a row; the dose is still chosen per row,
        because doses are not comparable across methods, but the target never
        is.
      - It does not print a flip rate without `n_flippable`.  A rate over 6
        items and a rate over 394 are different measurements.
      - It does not fill a missing control with zero.  `has_control` says
        whether `effect` is controlled; where it is False the number is a
        bare delta-margin and must not be read against a controlled one.

    `effect` is `effect_toward_target`: the controlled delta-margin re-signed
    so HIGHER is better for whichever target the row is about.  That is what
    makes a use_context row and a use_parametric row comparable.
    `conflict_specific` is the same quantity minus what the method did to the
    SAME items under a supporting passage - the benchmark's actual steering
    question.
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
                     "auroc_base": d.get("base_rate_auroc", np.nan),
                     "auroc_over_readout": d.get("auroc_over_readout", np.nan),
                     "auroc_raw": d.get("auroc_raw", np.nan),
                     "auroc_corrected": d.get("auroc_corrected", np.nan),
                     "accuracy": d.get("accuracy", np.nan),
                     "acc_majority": d.get("majority_accuracy", np.nan),
                     "n": d.get("n_instances", d.get("n_items")),
                     "note": CONTROL_NOTES.get(m, "")})

    # steering: one row per (method, target), at that arm's best dose on the
    # conflict condition
    if steering_df is not None and len(steering_df):
        sdf = steering_df
        if "condition" in sdf:
            conf = sdf[sdf.condition == conflict_condition]
            sdf = conf if len(conf) else sdf
        rank = sdf.get("effect_toward_target")
        if rank is None:
            rank = sdf.get("d_margin_toward_target", pd.Series(np.nan,
                                                               index=sdf.index))
        sdf = sdf.assign(_rank=rank.fillna(
            sdf.get("d_margin_toward_target", pd.Series(np.nan,
                                                        index=sdf.index))))
        for (m, t), g in sdf.groupby(["method", "target"]):
            g = g[g._rank.notna()]
            if not len(g):
                continue
            best = g.loc[g._rank.idxmax()]
            rows.append({
                "method": m, "axis": "steering", "target": t,
                "effect": best.get("effect_toward_target", np.nan),
                "conflict_specific": best.get("conflict_specific_effect",
                                              np.nan),
                "flip_rate": best.get("flip_rate", np.nan),
                "flip_flippable": best.get("flip_rate_flippable", np.nan),
                "n_flippable": best.get("n_flippable", np.nan),
                "flip_rate_raw": best.get("flip_rate_raw", np.nan),
                "flip_rate_corrected": best.get("flip_rate_corrected", np.nan),
                "has_control": bool(best.get("has_control", False)),
                "factor": best.get("factor"),
                "n": best.get("n"),
                "note": CONTROL_NOTES.get(m, "")})

    df = pd.DataFrame(rows, columns=SIMPLE_COLUMNS)
    if df.empty:
        return df
    order = {"detection": 0, "steering": 1}
    df["_axis"] = df.axis.map(order)
    df["_rank"] = np.where(df.axis == "detection",
                           df.auroc.fillna(-1), df.effect.fillna(-1e9))
    df = (df.sort_values(["_axis", "target", "_rank"],
                         ascending=[True, True, False], na_position="first")
            .drop(columns=["_axis", "_rank"]).reset_index(drop=True))
    num = [c for c in SIMPLE_COLUMNS
           if c not in ("method", "axis", "target", "n", "n_flippable",
                        "has_control", "note")]
    df[num] = df[num].astype(float).round(3)
    return df
