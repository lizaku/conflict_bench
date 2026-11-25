"""Step 3 - contexts.  beliefs.jsonl -> contexts.jsonl   (no model)

Every context is a list of sentences, and every sentence is a seed cloze
template with a value filled in.  No language model writes a context and no
NLI model checks one: what a sentence asserts is known by construction.

Two kinds of sentence:

  carrier  states a value for a fact on the answer path:
           M's own answer `a` (supports), another value `b` (contradicts),
           or any value when M does not know the fact (novel)
  filler   states another subject's fact that M KNOWS, with M's own answer -
           so a filler always supports its own fact and says nothing about q.
           (A filler stated with the Wikidata answer could contradict M by
           accident and create an unlabelled conflict.)

Fillers are drawn once per fact and reused by every recipe of that fact, so
instances of one fact differ only in the carrier.

`label()` is the whole labelling logic.  RECIPES below is the coverage table
of the taxonomy: one function per cell.
"""
import hashlib
import random
from collections import Counter, defaultdict

from conflict_bench.construct import match
from conflict_bench.construct.common import model_dir, write_jsonl
from conflict_bench.construct.seed import load_relations

#: made-up entities for the invented-bridge recipes ("Velorix was founded by
#: Marie Curie"). Hop 1 is novel by construction: no model can know them.
INVENTED = ["Velorix", "Quandrel", "Zemvara", "Tolbrick", "Oskerin", "Yadwell",
            "Brimtov", "Calvexa", "Durnholt", "Eskavir", "Fennoch", "Glimvarr",
            "Hobrelt", "Ilverak", "Jostrand", "Kelmora", "Lunvesk", "Morvath",
            "Nelquist", "Ostrevan", "Pavrion", "Quillomar", "Rethvik", "Sulmarin",
            "Tavrock", "Ulvenor", "Varnesk", "Wexlorn", "Xandivar", "Yorvelt"]

CORE = ["sup", "con", "irr", "nov"]
ALL_RECIPES = CORE + [
    "2hop-sup-con", "2hop-sup-sup", "2hop-con-sup", "2hop-nov-con", "2hop-nov-sup",
    "two-agree-disagree", "two-disagree-disagree",
    "off-path", "unrelated", "extra-true", "pos-first", "pos-last"]


def label(known: bool, gives_answer: bool, compatible: bool) -> str:
    """One fact, one context -> one of the four labels.

    known         condition 1: a = M(q) is stable and confident
    gives_answer  condition 2: c asserts b as an answer to q
    compatible    condition 3 negated: a and b can both be true
    """
    if not gives_answer:
        return "irrelevant"     # c says nothing about the fact
    if not known:
        return "novel"          # no a, so nothing to contradict
    if compatible:
        return "supports"
    return "contradicts"        # the only conflict


def statement(fact, value) -> str:
    return fact["cloze"].replace("____", value)


def _cap(s):
    return s[:1].upper() + s[1:]


class Builder:
    def __init__(self, facts, beliefs, pools, cfg, model_name):
        self.cfg = cfg
        self.ccfg = cfg.get("contexts", {})
        self.model_name = model_name
        self.seed = cfg.get("seed", 0)
        self.rels = load_relations()
        self.pools = pools
        self.B = {b["fact_id"]: b for b in beliefs}
        self.F = {f["fact_id"]: f for f in facts if f["fact_id"] in self.B}
        self.formats = self.ccfg.get("formats", ["short", "cloze"])
        self.recipes = set(self.ccfg.get("recipes") or ALL_RECIPES)
        self.out, self.rejected = [], []
        self._names = {}
        # facts M knows with an answer we can write down: the filler bank,
        # and the only facts that can be supported or contradicted
        self.known = sorted(fid for fid, b in self.B.items()
                            if b["state"] == "known" and b["a_qid"]
                            and fid in self.F)
        self.by_subject, self.by_answer = defaultdict(list), defaultdict(list)
        for fid in self.known:
            self.by_subject[self.F[fid]["subject_qid"]].append(fid)
            self.by_answer[self.F[fid]["gold_qid"]].append(fid)
        self.contradictable = [fid for fid in self.known
                               if self.can_contradict(fid)]

    # ------------------------------------------------------------ helpers
    def rng(self, *key):
        return random.Random(f"{self.seed}|" + "|".join(map(str, key)))

    def reject(self, fid, reason):
        self.rejected.append({"step": "contexts", "fact_id": fid, "reason": reason})

    def names(self, rel, qid):
        key = (rel, qid)
        if key not in self._names:
            e = self.pools[rel][qid]
            self._names[key] = match.names(e["name"], e["aliases"])
        return self._names[key]

    def can_contradict(self, fid) -> bool:
        """A swap on this fact is a guaranteed contradiction for M."""
        return (fid in self.F and self.F[fid]["single_valued"]
                and self.B[fid]["a_qid"] is not None
                and self.B[fid]["state"] in ("known", "uncertain"))

    def pick_b(self, fid, rng, exclude=()):
        """A value M's answer cannot be true together with: same relation,
        not `a`, never produced by M for this question, no name overlap."""
        f, b = self.F[fid], self.B[fid]
        a_names = self.names(f["relation"], b["a_qid"])
        subject = [match.norm(f["subject"])]
        cands = []
        for qid, e in sorted(self.pools[f["relation"]].items()):
            if qid == b["a_qid"] or qid in b["sampled_qids"] or qid in exclude:
                continue
            ns = self.names(f["relation"], qid)
            if (match.has(e["name"], a_names) or match.has(b["a"], ns)
                    or match.has(f["subject"], ns) or match.has(e["name"], subject)):
                continue
            cands.append(qid)
        return rng.choice(cands) if cands else None

    def pick_fillers(self, fid, n, avoid, rng):
        """`n` known facts of other subjects and other relations whose
        sentence mentions none of the `avoid` names."""
        f = self.F[fid]
        picked, subjects = [], {f["subject_qid"]}
        pool = [g for g in self.known if self.F[g]["relation"] != f["relation"]]
        rng.shuffle(pool)
        for g in pool:
            G = self.F[g]
            if G["subject_qid"] in subjects:
                continue
            if match.has(statement(G, self.B[g]["a"]), avoid):
                continue
            picked.append(g)
            subjects.add(G["subject_qid"])
            if len(picked) == n:
                return picked
        return None

    # ------------------------------------------------------------ pieces
    def carrier(self, fid, qid, on_path=True):
        """(chunk, path entry) for stating value `qid` of fact `fid`."""
        f, b = self.F[fid], self.B[fid]
        value = self.pools[f["relation"]][qid]["name"]
        entry = self.path_entry(fid, qid)
        chunk = {"text": statement(f, value), "role": "carrier", "fact_id": fid,
                 "asserts": value, "on_path": on_path, "label": entry["label"],
                 "label_if_known": entry["label_if_known"]}
        return chunk, entry

    def filler(self, gid):
        a = self.B[gid]["a"]
        return {"text": statement(self.F[gid], a), "role": "filler",
                "fact_id": gid, "asserts": a, "on_path": False,
                "label": "supports", "label_if_known": None}

    def path_entry(self, fid, qid):
        """The label of one path fact given the value (or None) c states."""
        f, b = self.F[fid], self.B[fid]
        known = b["state"] == "known"
        gives = qid is not None
        compatible = (qid == b["a_qid"]) if gives and b["a_qid"] else None
        would = None
        if b["state"] == "uncertain" and b["a_qid"]:
            # what the label would be if M's weak answer counted as knowledge;
            # uncertain facts are reported separately, with this label
            would = label(True, gives, bool(compatible))
        return {"fact_id": fid, "relation": f["relation"], "state": b["state"],
                "a": b["a"],
                "asserted": self.pools[f["relation"]][qid]["name"] if gives else None,
                "known": known, "gives_answer": gives, "compatible": compatible,
                "label": label(known, gives, bool(compatible)),
                "label_if_known": would}

    def emit(self, recipe, fid, stems, passages, path, targets, answers,
             features, contrast=None):
        """One instance per output format."""
        f = self.F[fid]
        chunks = [c for _, cs in passages for c in cs]
        on_path_ids = {p["fact_id"] for p in path}
        eff = lambda x: x["label_if_known"] or x["label"]
        feats = {"k": len(path), "structure": "1 hop" if len(path) == 1
                 else f"{len(path)} hops", "self_consistency": "consistent",
                 "location": None, "position": None, "repetition": 1}
        feats.update(features or {})
        labels = {
            "conflict_q": int(any(p["label"] == "contradicts" for p in path)),
            "conflict_global": int(any(c["label"] == "contradicts" for c in chunks)),
            "chunk_asserts_b": [int(c["label"] == "contradicts"
                                    and c["fact_id"] in on_path_ids)
                                for c in chunks],
            # the same three under "uncertain counts as known"
            "conflict_q_if_known": int(any(eff(p) == "contradicts" for p in path)),
            "conflict_global_if_known": int(any(eff(c) == "contradicts"
                                                for c in chunks)),
            "chunk_asserts_b_if_known": [int(eff(c) == "contradicts"
                                             and c["fact_id"] in on_path_ids)
                                         for c in chunks]}
        for fmt in self.formats:
            stem = stems.get(fmt)
            if not stem:
                continue
            key = f"{self.model_name}|{fid}|{recipe}|{fmt}"
            self.out.append({
                "instance_id": hashlib.sha1(key.encode()).hexdigest()[:10],
                "model": self.model_name, "fold": f["fold"], "group": fid,
                "recipe": recipe, "format": fmt, "relation": f["relation"],
                "state": path[-1]["state"],
                "has_uncertain": any(p["state"] == "uncertain" for p in path),
                "question": stem,
                "passages": [{"id": pid, "chunks": cs} for pid, cs in passages],
                "path": path, "answers": answers, "contrast": contrast,
                "labels": labels, "features": feats, "targets": targets})

    # ------------------------------------------------------------ recipes
    def one_hop(self, fid):
        f, b = self.F[fid], self.B[fid]
        rel = f["relation"]
        rng = self.rng(fid)
        stems = {"short": f["question"], "cloze": f["cloze_stem"]}
        want = self.recipes

        if b["state"] == "unknown":
            # any value is new to M; half the time the true one, half a swap,
            # so "is the claim false in the world" cannot separate con from nov
            cands = [q for q in sorted(self.pools[rel])
                     if q != f["gold_qid"] and q not in b["sampled_qids"]
                     and not match.has(f["subject"], self.names(rel, q))]
            v = f["gold_qid"] if (rng.random() < 0.5 or not cands) \
                else rng.choice(cands)
            if v not in self.pools[rel]:
                return self.reject(fid, "gold answer not in the answer pool")
            v_name = self.pools[rel][v]["name"]
            answers = {v_name: self.names(rel, v)}
            fill = self.pick_fillers(fid, 4, answers[v_name] + [match.norm(f["subject"])], rng)
            if not fill:
                return self.reject(fid, "not enough fillers")
            F1, F2, F3, F4 = map(self.filler, fill)
            if "nov" in want:
                c, p = self.carrier(fid, v)
                self.emit("nov", fid, stems, [("A", [F1, c, F2, F3])], [p],
                          {"ctx": v_name, "mem": v_name}, answers,
                          {"location": "on answer path", "position": "middle",
                           "claim_is_gold": v == f["gold_qid"]})
            if "irr" in want:
                self.emit("irr", fid, stems, [("A", [F1, F4, F2, F3])],
                          [self.path_entry(fid, None)],
                          {"ctx": None, "mem": None}, answers, {})
            return

        if not b["a_qid"]:
            return self.reject(fid, "stable answer outside the answer pool"
                               if b["state"] == "known"
                               else "uncertain, modal answer outside the answer pool")
        a_q, a = b["a_qid"], b["a"]
        b_q = self.pick_b(fid, rng)
        if not b_q:
            return self.reject(fid, "no contradicting value available")
        b_name = self.pools[rel][b_q]["name"]
        answers = {a: self.names(rel, a_q), b_name: self.names(rel, b_q)}
        avoid = answers[a] + answers[b_name] + [match.norm(f["subject"])]
        fill = self.pick_fillers(fid, 4, avoid, rng)
        if not fill:
            return self.reject(fid, "not enough fillers")
        F1, F2, F3, F4 = map(self.filler, fill)
        sup_c, sup_p = self.carrier(fid, a_q)
        con_c, con_p = self.carrier(fid, b_q)
        contrast = {"a": a, "b": b_name}
        on = {"location": "on answer path"}
        T_sup, T_con = {"ctx": a, "mem": a}, {"ctx": b_name, "mem": a}

        def emit(recipe, passages, path, targets, feats, ans=answers):
            if recipe in want:
                self.emit(recipe, fid, stems, passages, path, targets, ans,
                          feats, contrast)

        emit("sup", [("A", [F1, sup_c, F2, F3])], [sup_p], T_sup, {})
        emit("con", [("A", [F1, con_c, F2, F3])], [con_p], T_con,
             dict(on, position="middle"))
        emit("irr", [("A", [F1, F4, F2, F3])], [self.path_entry(fid, None)],
             {"ctx": None, "mem": a}, {})
        if b["state"] != "known":
            return                       # slices are built on known facts only

        emit("pos-first", [("A", [con_c, F1, F2, F3])], [con_p], T_con,
             dict(on, position="first"))
        emit("pos-last", [("A", [F1, F2, F3, con_c])], [con_p], T_con,
             dict(on, position="last"))

        # ---- context self-consistency: two passages
        a_first = rng.random() < 0.5
        x, y = (sup_c, con_c) if a_first else (con_c, sup_c)
        emit("two-agree-disagree",
             [("A", [F1, x, F2]), ("B", [F3, y, F4])], [con_p],
             {"ctx": None, "mem": a,
              "span": {"A": x["asserts"], "B": y["asserts"]}},
             dict(on, self_consistency="one passage agrees with M, one doesn't"))
        b2_q = self.pick_b(fid, rng, exclude={b_q})
        if b2_q and "two-disagree-disagree" in want:
            b2 = self.pools[rel][b2_q]["name"]
            ans3 = dict(answers, **{b2: self.names(rel, b2_q)})
            if not any(match.has(c["text"], ans3[b2]) for c in (F1, F2, F3, F4)):
                c2, _ = self.carrier(fid, b2_q)
                emit("two-disagree-disagree",
                     [("A", [F1, con_c, F2]), ("B", [F3, c2, F4])], [con_p],
                     {"ctx": None, "mem": a, "span": {"A": b_name, "B": b2}},
                     dict(on, self_consistency="passages disagree with M and each other"),
                     ans3)

        # ---- where the contradiction is: a second claim after a supported q
        same = [g for g in self.by_subject[f["subject_qid"]] if g != fid]
        off = [g for g in same if self.can_contradict(g)]
        if off and "off-path" in want:
            g = rng.choice(off)
            gb = self.pick_b(g, rng)
            if gb:
                c, _ = self.carrier(g, gb, on_path=False)
                if not match.has(c["text"], answers[b_name]):
                    emit("off-path", [("A", [F1, sup_c, F2, c])], [sup_p], T_sup,
                         {"location": "same entity, off path"})
        if same and "extra-true" in want:
            g = rng.choice(same)
            c, _ = self.carrier(g, self.B[g]["a_qid"], on_path=False)
            if not match.has(c["text"], answers[b_name]):
                emit("extra-true", [("A", [F1, sup_c, F2, c])], [sup_p], T_sup, {})
        if "unrelated" in want:
            others = [g for g in self.contradictable
                      if self.F[g]["subject_qid"] != f["subject_qid"]
                      and self.F[g]["relation"] != rel]
            for g in rng.sample(others, min(5, len(others))):
                gb = self.pick_b(g, rng)
                c = self.carrier(g, gb, on_path=False)[0] if gb else None
                if c and not match.has(c["text"], avoid):
                    emit("unrelated", [("A", [F1, sup_c, F2, c])], [sup_p],
                         T_sup, {"location": "unrelated"})
                    break

        self.two_hop(fid, fill, answers, contrast, rng)

    def two_hop(self, fid, fill, answers, contrast, rng):
        """f2 = `fid` is the last hop; hop 1 is a real seed fact or invented."""
        f2, b2 = self.F[fid], self.B[fid]
        want = self.recipes
        if not any(r in want for r in ALL_RECIPES if r.startswith("2hop")):
            return
        s2 = f2["subject"]
        a, b_name = contrast["a"], contrast["b"]
        F1, F2, F3 = map(self.filler, fill[:3])
        sup_c, sup_p = self.carrier(fid, b2["a_qid"])
        b_q = next(q for q, e in self.pools[f2["relation"]].items()
                   if e["name"] == b_name)
        con_c, con_p = self.carrier(fid, b_q)

        def stems_for(np):
            """The question of f2 with its subject replaced by a description."""
            return {"short": _cap(f2["question"].replace(s2, np))
                    if s2 in f2["question"] else None,
                    "cloze": _cap(f2["cloze_stem"].replace(s2, np))
                    if f2["cloze_stem"] and s2 in f2["cloze_stem"] else None}

        two = {"location": "on answer path"}

        # ---- real chains: a seed fact whose answer is the subject of f2
        hop1 = [g for g in self.by_answer[f2["subject_qid"]]
                if self.B[g]["a_equals_gold"]
                and self.rels[self.F[g]["relation"]].get("np")
                and self.F[g]["subject_qid"] != f2["subject_qid"]]
        if hop1:
            g = rng.choice(sorted(hop1))
            G = self.F[g]
            stems = stems_for(self.rels[G["relation"]]["np"].format(s=G["subject"]))
            c1, p1 = self.carrier(g, self.B[g]["a_qid"])
            if "2hop-sup-con" in want:
                self.emit("2hop-sup-con", fid, stems, [("A", [c1, con_c, F1, F2])],
                          [p1, con_p], {"ctx": b_name, "mem": a}, answers, two, contrast)
            if "2hop-sup-sup" in want:
                self.emit("2hop-sup-sup", fid, stems, [("A", [c1, sup_c, F1, F2])],
                          [p1, sup_p], {"ctx": a, "mem": a}, answers, {}, contrast)
            # conflict on hop 1: the context names another bridge entity, for
            # which M knows the hop-2 fact, so "follow the context" is defined
            if "2hop-con-sup" in want and self.can_contradict(g):
                alts = []
                for h in self.known:
                    H = self.F[h]
                    if (H["relation"] == f2["relation"] and h != fid
                            and H["subject_qid"] in self.pools[G["relation"]]
                            and H["subject_qid"] not in self.B[g]["sampled_qids"]
                            and self.B[h]["a_qid"] != b2["a_qid"]):
                        alts.append(h)
                if alts:
                    h = rng.choice(sorted(alts))
                    t_ctx = self.B[h]["a"]
                    c_alt, p_alt = self.carrier(g, self.F[h]["subject_qid"])
                    ans = {a: answers[a],
                           t_ctx: self.names(f2["relation"], self.B[h]["a_qid"])}
                    if p_alt["label"] == "contradicts" and t_ctx != a:
                        self.emit("2hop-con-sup", fid, stems,
                                  [("A", [c_alt, F1, F2, F3])],
                                  [p_alt, self.path_entry(fid, None)],
                                  {"ctx": t_ctx, "mem": a}, ans, two,
                                  {"a": a, "b": t_ctx})

        # ---- invented bridge: hop 1 exists only in the context
        if "2hop-nov-con" in want or "2hop-nov-sup" in want:
            X = INVENTED[rng.randrange(len(INVENTED))]
            person = self.rels[f2["relation"]].get("person")
            text = f"{X} was founded by {s2}." if person else f"{X} is named after {s2}."
            np = f"the founder of {X}" if person else f"the namesake of {X}"
            bid = f"BRIDGE:{X}:{fid}"
            bridge = {"text": text, "role": "carrier", "fact_id": bid,
                      "asserts": s2, "on_path": True, "label": "novel",
                      "label_if_known": None}
            p1 = {"fact_id": bid, "relation": "BRIDGE", "state": "unknown",
                  "a": None, "asserted": s2, "known": False, "gives_answer": True,
                  "compatible": None, "label": "novel", "label_if_known": None}
            stems = stems_for(np)
            nov = {"no_context_fails": True}
            if "2hop-nov-con" in want:
                self.emit("2hop-nov-con", fid, stems, [("A", [bridge, con_c, F1, F2])],
                          [p1, con_p], {"ctx": b_name, "mem": a}, answers,
                          dict(two, **nov), contrast)
            if "2hop-nov-sup" in want:
                self.emit("2hop-nov-sup", fid, stems, [("A", [bridge, sup_c, F1, F2])],
                          [p1, sup_p], {"ctx": a, "mem": a}, answers, nov, contrast)

    # ------------------------------------------------------------ driver
    def build(self):
        for fid in sorted(self.F):
            if not self.F[fid]["single_valued"]:
                self.reject(fid, "multi-valued relation: filler / hop 1 only")
                continue
            self.one_hop(fid)
        return self.out, self.rejected


def run(cfg, facts, beliefs, pools, model_name):
    out, rejected = Builder(facts, beliefs, pools, cfg, model_name).build()
    d = model_dir(cfg)
    write_jsonl(d / "contexts.jsonl", out)
    write_jsonl(d / "rejected.jsonl", rejected)
    print(f"[contexts] {len(out)} instances from "
          f"{len({i['group'] for i in out})} facts -> {d / 'contexts.jsonl'}")
    print("[contexts] per recipe:", dict(sorted(Counter(
        i["recipe"] for i in out).items())))
    print("[contexts] rejected:", dict(Counter(r["reason"] for r in rejected)))
    return out
