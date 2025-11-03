# retriever.py
from __future__ import annotations
import json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------- Data types ----------
@dataclass
class Doc:
    id: str
    text: str
    metadata: Dict

@dataclass
class Hit:
    doc: Doc
    score: float
    scores: Dict[str, float]  # {"vec": ..., "kw": ...}

# ---------- Text utils ----------
_word = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in _word.findall(s or "")]

# ---------- Corpus Index ----------
class CorpusIndex:
    def __init__(
        self,
        docs: List[Doc],
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.docs = docs
        self.N = len(docs)

        # Keyword side: build IDF over document terms
        self.doc_tokens: List[List[str]] = [tokenize(d.text) for d in docs]
        df: Dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.idf: Dict[str, float] = {
            t: math.log((1 + self.N) / (1 + c)) + 1.0 for t, c in df.items()
        }

        # Vector side: precompute embeddings if possible
        self.embed_fn = embed_fn
        if embed_fn is not None:
            emb = np.asarray(embed_fn([d.text for d in docs]), dtype="float32")
            # Normalize for cosine
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            self.doc_vecs = emb / norms
        else:
            self.doc_vecs = None  # keyword-only fallback

    # ----- scoring -----
    def _kw_score(self, query: str, toks: Optional[List[str]] = None) -> np.ndarray:
        q = tokenize(query) if toks is None else toks
        idf = self.idf
        # Simple sum of IDF for matched terms (works well for titles/fields)
        scores = np.zeros(self.N, dtype="float32")
        for i, dtoks in enumerate(self.doc_tokens):
            s = 0.0
            dtset = set(dtoks)
            for t in q:
                if t in dtset:
                    s += idf.get(t, 0.0)
            scores[i] = s
        # L2 normalize so it can be combined with vector scores
        if scores.max() > 0:
            scores = scores / (np.linalg.norm(scores) + 1e-12)
        return scores

    def _vec_score(self, query: str) -> Optional[np.ndarray]:
        if self.doc_vecs is None or self.embed_fn is None:
            return None
        qv = np.asarray(self.embed_fn([query])[0], dtype="float32")
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        sims = (self.doc_vecs @ qv).astype("float32")  # cosine similarity
        # Shift to positive & normalize
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)
        return sims

    # ----- public search -----
    def search(
        self,
        query: str,
        top_k: int = 8,
        alpha_vec: float = 0.6,  # weight for vector score
        alpha_kw: float = 0.4,   # weight for keyword score
        filters: Optional[Dict] = None,
    ) -> List[Hit]:
        filters = filters or {}

        # Pre-filter candidates by metadata if filters were provided
        mask = np.ones(self.N, dtype=bool)
        if filters:
            for i, d in enumerate(self.docs):
                keep = True
                for k, v in filters.items():
                    if v is None:
                        continue
                    mv = d.metadata.get(k)
                    # basic equality / membership / threshold support
                    if isinstance(v, (list, tuple, set)):
                        keep = keep and (mv in v)
                    elif isinstance(v, dict):
                        # numeric thresholds: {"gte": 3.5} or {"lte": 80}
                        if "gte" in v and mv is not None:
                            keep = keep and (float(mv) >= float(v["gte"]))
                        if "lte" in v and mv is not None:
                            keep = keep and (float(mv) <= float(v["lte"]))
                    else:
                        keep = keep and (mv == v)
                mask[i] = keep

        # Compute scores
        kw = self._kw_score(query)
        vec = self._vec_score(query)

        # Combine
        if vec is None:
            combo = kw
            alpha_v, alpha_k = 0.0, 1.0
        else:
            combo = alpha_vec * vec + alpha_kw * kw
            alpha_v, alpha_k = alpha_vec, alpha_kw

        # Mask out filtered rows
        combo_masked = np.where(mask, combo, -1.0)

        # Top-k
        idx = np.argsort(-combo_masked)[:top_k]
        hits: List[Hit] = []
        for i in idx:
            if combo_masked[i] <= -0.5:  # all filtered out
                continue
            scores = {"vec": float(vec[i]) if vec is not None else 0.0,
                      "kw": float(kw[i])}
            hits.append(Hit(self.docs[i], float(combo[i]), scores))
        return hits

# ---------- Loader ----------
def load_corpus_from_schema(schema_path: str | Path) -> List[Doc]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    jsonl_path = schema["files"]["jsonl_chunks"]
    docs: List[Doc] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Doc(id=str(rec["id"]), text=rec["text"], metadata=rec["metadata"]))
    return docs

# ---------- Context builder ----------
def format_hits_as_context(
    hits: List[Hit],
    max_chars: int = 4000,
    include_scores: bool = True,
    show_all_metadata: bool = False,
) -> str:
    """
    Build a richer context block for the LLM.
    - Adds key stats (female %, opinion score, student count, career prospects).
    - Optionally includes retrieval scores (vec/kw/combo) for transparency/debugging.
    - Can include *all* metadata (except the ones already shown) if show_all_metadata=True.
    - Still truncates to max_chars for safety.
    """

    def _fmt_num(x):
        # Nice, compact number formatting
        if x is None:
            return "N/A"
        try:
            xv = float(x)
        except Exception:
            return str(x)
        # choose precision based on magnitude
        if abs(xv) >= 1000:
            return f"{xv:,.0f}"
        if abs(xv) >= 100:
            return f"{xv:.0f}"
        if abs(xv) >= 10:
            return f"{xv:.1f}"
        return f"{xv:.2f}"

    def _append_stats(lines, m: Dict):
        opinion = m.get("overall_opinion_score")
        female_pct = m.get("first_year_female_pct")
        students = m.get("student_count")
        has_career = m.get("has_career_prospects")

        stats = []
        if opinion is not None:
            stats.append(f"Overall opinion score: {_fmt_num(opinion)}")
        if female_pct is not None:
            stats.append(f"First-year female %: {_fmt_num(female_pct)}")
        if students is not None:
            # try to cast to int if it looks integral
            try:
                si = int(float(students))
                stats.append(f"Student count: {si}")
            except Exception:
                stats.append(f"Student count: {_fmt_num(students)}")
        if has_career is True:
            stats.append("Career prospects: available")
        elif has_career is False:
            stats.append("Career prospects: not available")

        if stats:
            lines.append("Stats: " + " | ".join(stats))

    def _append_all_metadata(lines, m: Dict):
        # Show everything except the fields already rendered above
        exclude = {
            "program_id",
            "program_name",
            "program_name_en",
            "program_type",
            "program_variant",
            "cluster",
            "url",
            "overall_opinion_score",
            "first_year_female_pct",
            "student_count",
            "has_career_prospects",
        }
        extras = []
        for k, v in m.items():
            if k in exclude:
                continue
            # Render numbers nicely; otherwise str()
            try:
                vv = _fmt_num(v) if isinstance(v, (int, float, str)) else str(v)
            except Exception:
                vv = str(v)
            extras.append(f"{k}: {vv}")
        if extras:
            lines.append("Metadata:")
            for row in extras:
                lines.append(f"  - {row}")

    lines: List[str] = []
    for h in hits:
        m = h.doc.metadata or {}
        program_id = m.get("program_id") or h.doc.id

        # Header
        lines.append("-----")
        lines.append(f"Source Program ID: {program_id}")
        lines.append(
            "Program: "
            f"{m.get('program_name')} / {m.get('program_name_en')}"
        )
        lines.append(
            "Type: "
            f"{m.get('program_type')} | Variant: {m.get('program_variant')} | Cluster: {m.get('cluster')}"
        )
        lines.append(f"URL: {m.get('url')}")

        # Key stats
        _append_stats(lines, m)

        # Optional retrieval scores (useful for debugging or tie-break logic inside the prompt)
        if include_scores:
            s_vec = h.scores.get("vec", 0.0)
            s_kw = h.scores.get("kw", 0.0)
            lines.append(f"Retriever scores — combo: {_fmt_num(h.score)}, vec: {_fmt_num(s_vec)}, kw: {_fmt_num(s_kw)}")

        # Optional: dump remaining metadata for richer grounding
        if show_all_metadata:
            _append_all_metadata(lines, m)

        # Text snippet
        snippet = (h.doc.text or "").strip()
        # Give the model a generous snippet but let overall max_chars handle safety
        lines.append("TEXT:")
        lines.append(snippet[:1800])  # per-doc cap to avoid one long doc eating the whole budget
        lines.append("")  # blank line between hits

    ctx = "\n".join(lines).strip()
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars].rstrip() + "\n…"
    return ctx


