"""The N/S/C/R margin table - the design the whole benchmark is built on.

Every condition is actually scored, once, here:

    N  no passage          what the model believes unprompted
    S  supporting passage  passage agrees with the parametric answer
    C  conflicting passage counterfactual passage - the arbitration condition
    R  random passage      irrelevant passage of the same shape: the FORMAT
                           baseline, and nothing else

For one (item, condition) the table holds a `ScoredCandidates`: the
teacher-forced log-probs of the true and counterfactual answers, and

    margin = logp_true - logp_cf        > 0 parametric wins, < 0 context wins

R exists because a bare margin under C confounds two things: how much the
counterfactual passage moved the model, and how the passage *template* moved
the two answer strings relative to each other before any conflict was
involved.  R has the second effect and not the first, so

    corrected(C) = margin(C) - margin(R)

is the part attributable to the conflict.  Both numbers are reported
everywhere - the raw margin is the quantity actually observed, the corrected
one is the quantity that is supposed to mean something, and a gap between the
two conclusions is itself the finding.

The table is shared (attached as `model.margins`), cached and persisted, so
the R margin is computed once per item rather than once per method that
happens to want it.
"""
import json
from pathlib import Path

from conflict_bench.core.types import Condition, ScoredCandidates
from conflict_bench.core.prompts import build_prompt

ALL_CONDITIONS = [Condition.NORMAL, Condition.SUPPORTING,
                  Condition.CONFLICTING, Condition.IRRELEVANT]


def as_condition(c):
    return c if isinstance(c, Condition) else Condition(str(c).upper()[:1])


class MarginTable:
    def __init__(self, model, path=None, reference=Condition.IRRELEVANT):
        self.model = model
        self.reference = as_condition(reference)
        self.correct = True   # set False when R is not in the run's conditions
        self.path = Path(path) if path else None
        self._cache: dict[tuple, ScoredCandidates] = {}
        self.n_scored = 0
        if self.path and self.path.exists():
            self.load()

    # ---------- persistence ----------
    def load(self):
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            sc = ScoredCandidates(d["item_id"], Condition(d["condition"]),
                                  d["logp_true"], d["logp_cf"],
                                  d.get("logp_distractors", {}))
            self._cache[(sc.item_id, sc.condition.value)] = sc

    def flush(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for sc in self._cache.values():
                f.write(json.dumps({"item_id": sc.item_id,
                                    "condition": sc.condition.value,
                                    "logp_true": sc.logp_true,
                                    "logp_cf": sc.logp_cf,
                                    "margin": sc.margin}) + "\n")

    # ---------- scoring ----------
    def scored(self, item, condition) -> ScoredCandidates:
        condition = as_condition(condition)
        key = (item.item_id, condition.value)
        if key not in self._cache:
            p = build_prompt(item, condition)
            self._cache[key] = ScoredCandidates(
                item_id=item.item_id, condition=condition,
                logp_true=self.model.logp_continuation(p, " " + item.true_answer),
                logp_cf=self.model.logp_continuation(
                    p, " " + item.counterfactual_answer))
            self.n_scored += 1
        return self._cache[key]

    def margin(self, item, condition) -> float:
        """Raw margin: what was actually observed under this condition."""
        return self.scored(item, condition).margin

    def offset(self, item) -> float:
        """The format baseline: the margin under the reference condition (R).

        Zero when the reference condition is not part of this run - a run
        without R reports raw margins only rather than silently correcting
        against a condition it never scored.
        """
        if not self.correct:
            return 0.0
        return self.margin(item, self.reference)

    def corrected(self, item, condition) -> float:
        """Margin with the format baseline removed."""
        condition = as_condition(condition)
        if self.correct and condition == self.reference:
            return 0.0
        return self.margin(item, condition) - self.offset(item)

    def score_all(self, items, conditions=None):
        """Run the full design. Returns the flat list of ScoredCandidates."""
        conditions = [as_condition(c) for c in (conditions or ALL_CONDITIONS)]
        return [self.scored(it, c) for it in items for c in conditions]

    # ---------- reporting ----------
    def rows(self, items, conditions=None, labels=None):
        """One row per item: every condition raw, and every condition
        corrected against the reference. The design matrix, wide."""
        conditions = [as_condition(c) for c in (conditions or ALL_CONDITIONS)]
        label_of = dict(zip((it.item_id for it in items), labels or []))
        out = []
        for it in items:
            row = {"item_id": it.item_id, "relation": it.relation}
            if labels is not None:
                row["label"] = label_of.get(it.item_id)
            for c in conditions:
                sc = self.scored(it, c)
                row[f"logp_true_{c.value}"] = sc.logp_true
                row[f"logp_cf_{c.value}"] = sc.logp_cf
                row[f"margin_{c.value}"] = sc.margin
            if self.correct and self.reference in conditions:
                for c in conditions:
                    if c != self.reference:
                        row[f"margin_{c.value}_corr"] = self.corrected(it, c)
            out.append(row)
        return out

    def stats(self):
        return {"scored_pairs": self.n_scored, "cached": len(self._cache),
                "reference": self.reference.value if self.correct else None,
                "format_correction": self.correct,
                "path": str(self.path) if self.path else None}


def condition_summary(rows, conditions=None, reference="R"):
    """Per-condition sanity table: is the manipulation doing anything?

    `context_win_rate` is the fraction of items whose margin is negative, i.e.
    where the counterfactual answer outscores the true one.  Under N and S it
    should be low (nothing is pulling toward the counterfactual); under C it is
    the whole effect; under R it is the format artifact the correction removes,
    and if it is far from the N rate then R is doing more than formatting and
    the correction is suspect.
    """
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows)
    conditions = [as_condition(c).value for c in (conditions or ALL_CONDITIONS)]
    ref = as_condition(reference).value
    out = []
    for c in conditions:
        col = f"margin_{c}"
        if col not in df:
            continue
        rec = {"condition": c, "n": int(df[col].notna().sum()),
               "mean_margin": float(df[col].mean()),
               "sd_margin": float(df[col].std()),
               "context_win_rate": float((df[col] < 0).mean())}
        corr = f"margin_{c}_corr"
        if corr in df:
            rec.update(mean_margin_corrected=float(df[corr].mean()),
                       sd_margin_corrected=float(df[corr].std()),
                       context_win_rate_corrected=float((df[corr] < 0).mean()),
                       mean_shift_vs_raw=float(df[corr].mean() - df[col].mean()))
        elif c == ref:
            rec["note"] = "reference condition (the correction itself)"
        if "label" in df and df["label"].notna().any():
            rec["auroc_vs_behavioural_label"] = _safe_auroc(
                -df[col], df["label"])
            if corr in df:
                rec["auroc_corrected"] = _safe_auroc(-df[corr], df["label"])
        out.append(rec)
    return pd.DataFrame(out)


def _safe_auroc(score, label):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    label = np.asarray(label, dtype=float)
    ok = ~np.isnan(label)
    if len(set(label[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(label[ok], np.asarray(score)[ok]))
