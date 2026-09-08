"""Dataset loaders -> list[Item].

Primary: the clustered ConFiQA cloze dataset, `data/ConfiQA_cloze_clustered/`,
one JSON file per Wikidata relation (P6.json, P17.json, ...), 40 relations /
6000 records.  Record schema (uniform across all files):

    orig_path, cf_path                 "[('Q235', 'P6', 'Q3027038')]"
    orig_path_labeled, cf_path_labeled "[('Monaco', 'head of government', ...)]"
    orig_triple, cf_triple             single-hop triple as a string
    orig_answer, cf_answer             "Didier Guillaume" / "John Swinney"
    orig_alias, cf_alias               list[str]
    question                           "Who is the head of government of Monaco?"
    orig_context, cf_context           full paragraph (the S and C passages)
    orig_context_piece, cf_context_piece   the single sentence that carries it
    cloze_template                     "The head of government of Monaco is ____."
    orig_statement, cf_statement       the filled cloze

Two things the raw files do not provide and this loader supplies:
  * `subject` - parsed out of orig_path_labeled (head of the first triple).
  * the **R passage** - no random/irrelevant context ships with the data, but
    the whole margin DV is format-corrected against R.  We synthesise it
    deterministically: each item borrows the orig_context of an item from a
    *different relation*, so R is a real paragraph of the same register that
    is topically irrelevant to the question.
"""
import ast
import json
import random
from collections import defaultdict
from pathlib import Path

from conflict_bench.core.types import Item

DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "ConfiQA_cloze_clustered"


def _parse_path(raw):
    try:
        return ast.literal_eval(raw) if raw else []
    except (ValueError, SyntaxError):
        return []


def _subject(rec):
    """Head entity of the first labeled triple; empty string if unparsable."""
    for key in ("orig_path_labeled", "cf_path_labeled"):
        path = _parse_path(rec.get(key))
        if path and len(path[0]) >= 1:
            return str(path[0][0])
    return ""


def _relation_label(rec):
    path = _parse_path(rec.get("orig_path_labeled"))
    if path and len(path[0]) >= 2:
        return str(path[0][1])
    return ""


def _attach_random_passages(items, seed=0, key="R"):
    """Give every item an irrelevant passage drawn from a different relation.

    The R condition is the format baseline the margin is corrected against, so
    it has to be a passage of the same shape and length distribution as S/C -
    an empty string or a generic filler would not correct for the same thing.
    """
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for it in items:
        by_rel[it.relation].append(it)
    relations = sorted(by_rel)
    for it in items:
        others = [r for r in relations if r != it.relation]
        pool = by_rel[rng.choice(others)] if others else [
            o for o in items if o.subject != it.subject] or items
        other = rng.choice(pool)
        it.passages[key] = other.passages["S"]
        it.meta["random_passage_from"] = other.item_id
    return items


def load_confiqa_cloze_clusters(data_dir=DEFAULT_DATA_DIR, relations=None,
                                use_context_piece=False, seed=0,
                                min_alias_len=0, exclude_item_ids=None,
                                max_per_relation=None):
    """Load the clustered cloze dataset. Returns (items, dropped).

    use_context_piece: use the single carrier sentence instead of the full
        paragraph as the S/C passage (a much denser conflict signal per token;
        the full paragraph is the default because it is what ConFiQA evaluates
        on).
    min_alias_len: drop items whose cf answer is shorter than this many
        characters (the ConFiQA short-answer artifact); 0 disables.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("P*.json"))
    if not files:
        raise FileNotFoundError("no P*.json under " + str(data_dir))
    if relations:
        wanted = {str(r) for r in relations}
        files = [f for f in files if f.stem in wanted]
    exclude = set(exclude_item_ids or ())
    ctx_key = "context_piece" if use_context_piece else "context"

    items, dropped = [], []
    for f in files:
        recs = json.loads(f.read_text(encoding="utf-8"))
        if max_per_relation:
            recs = recs[:max_per_relation]
        for i, rec in enumerate(recs):
            iid = f.stem + ":" + str(i)
            if iid in exclude:
                dropped.append({"item_id": iid, "reasons": ["excluded"]})
                continue
            cf = rec["cf_answer"]
            if min_alias_len and len(cf.strip()) < min_alias_len:
                dropped.append({"item_id": iid, "reasons": ["short_cf_answer"],
                                "cf_answer": cf})
                continue
            items.append(Item(
                item_id=iid,
                relation=f.stem,
                cloze_template=rec["cloze_template"],
                subject=_subject(rec),
                true_answer=rec["orig_answer"],
                counterfactual_answer=cf,
                question=rec["question"],
                true_aliases=list(rec.get("orig_alias") or []),
                cf_aliases=list(rec.get("cf_alias") or []),
                passages={"S": rec["orig_" + ctx_key],
                          "C": rec["cf_" + ctx_key]},
                meta={"relation_label": _relation_label(rec),
                      "orig_triple": rec.get("orig_triple"),
                      "cf_triple": rec.get("cf_triple"),
                      "orig_statement": rec.get("orig_statement"),
                      "cf_statement": rec.get("cf_statement"),
                      "source_file": f.name, "source_index": i},
            ))
    _attach_random_passages(items, seed=seed)
    return items, dropped


# Back-compat alias: the config used to name this loader after the v3 pipeline.
load_wikidata_relations = load_confiqa_cloze_clusters


def load_confiqa(path, apply_qc=True, min_alias_len=3):
    """Raw ConFiQA-QA (the flat 13MB json) with the artifact filters:
       - drop counterfactual answers/aliases shorter than min_alias_len chars
       - drop items where the counterfactual context rewrites the SUBJECT
         entity (subject string missing from the cf context)
    Keep a log of what was dropped - that log IS the benchmark-audit paper."""
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    items, dropped = [], []
    for i, r in enumerate(recs):
        reasons = []
        subj = r.get("subject") or _subject(r)
        cf_answer = r.get("cf_answer", r.get("counterfactual_answer", ""))
        cf_context = r.get("cf_context", r.get("counterfactual_context", ""))
        if apply_qc:
            if len(cf_answer) < min_alias_len:
                reasons.append("short_cf_answer")
            if any(len(a) < min_alias_len for a in (r.get("cf_alias") or [])):
                reasons.append("short_alias")
            if subj and subj not in cf_context:
                reasons.append("subject_rewritten")
        if reasons:
            dropped.append({"idx": i, "reasons": reasons})
            continue
        items.append(Item(
            item_id="confiqa:" + str(i),
            relation=r.get("relation", "unknown"),
            cloze_template=r.get("cloze_template", ""),
            subject=subj,
            true_answer=r.get("orig_answer", r.get("answer", "")),
            counterfactual_answer=cf_answer,
            question=r["question"],
            true_aliases=list(r.get("orig_alias") or r.get("aliases") or []),
            cf_aliases=list(r.get("cf_alias") or []),
            passages={"S": r.get("orig_context", r.get("context", "")),
                      "C": cf_context},
        ))
    _attach_random_passages(items)
    return items, dropped


def first_token_collisions(items, tokenizer, answer_prefix=" "):
    """Item ids whose true and cf answers share a first token.

    CAD/CK-PLUG score a first-token margin, so a collision item has a
    structurally zero DV there.  The v3 pipeline dropped 198 such items; the
    exact set depends on the tokenizer, so it is computed per model rather
    than shipped as a static id list.
    """
    hits = []
    for it in items:
        t = tokenizer(answer_prefix + it.true_answer,
                      add_special_tokens=False).input_ids
        c = tokenizer(answer_prefix + it.counterfactual_answer,
                      add_special_tokens=False).input_ids
        collide = bool(t) and bool(c) and t[0] == c[0]
        it.meta["first_token_collision"] = collide
        if collide:
            hits.append(it.item_id)
    return hits
