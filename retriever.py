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
def format_hits_as_context(hits: List[Hit], max_chars: int = 2500) -> str:
    """Create the context snippet to stuff into the prompt, citing program_id."""
    lines = []
    for h in hits:
        m = h.doc.metadata
        program_id = m.get("program_id") or h.doc.id  # fallback to id if missing
        lines.append(
            "-----\n"
            f"Source Program ID: {program_id}\n"
            f"Program: {m.get('program_name')} / {m.get('program_name_en')}\n"
            f"Type: {m.get('program_type')} | Variant: {m.get('program_variant')} | Cluster: {m.get('cluster')}\n"
            f"URL: {m.get('url')}\n"
            f"TEXT:\n{h.doc.text[:1200].strip()}\n"
        )
    ctx = "\n".join(lines)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n…"
    return ctx

