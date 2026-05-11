from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def _chars(text: str) -> list[str]:
    return [char for char in str(text).strip() if not char.isspace()]


def candidate_aliases(candidate: dict[str, Any]) -> set[str]:
    aliases = {str(candidate.get("pt_name", "")).strip()}
    aliases.update(str(alias).strip() for alias in candidate.get("llt_names", []) if str(alias).strip())
    return {alias for alias in aliases if alias}


def exact_match_rank(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query = str(query).strip()
    exact: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for candidate in candidates:
        if query in candidate_aliases(candidate):
            exact.append(dict(candidate, match_type="exact", retrieval_score=1.0, prob_1=1.0))
        else:
            others.append(dict(candidate, match_type="none", retrieval_score=0.0, prob_1=0.0))
    return exact + others


class CharacterBM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized = [_chars(document) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.tokenized]
        df: dict[str, int] = defaultdict(int)
        for freqs in self.term_freqs:
            for token in freqs:
                df[token] += 1
        n_docs = len(self.tokenized)
        self.idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        query_terms = _chars(query)
        if not query_terms or not self.tokenized:
            return [0.0 for _ in self.tokenized]
        scores: list[float] = []
        for freqs, doc_len in zip(self.term_freqs, self.doc_lengths):
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-12))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1.0) / denom
            scores.append(score)
        return scores


def lexical_bm25_rank(query: str, candidates: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
    bm25 = CharacterBM25([str(candidate.get("text_b", "")) for candidate in candidates])
    return lexical_bm25_rank_indexed(query, bm25, candidates, top_k=top_k)


def build_bm25_index(candidates: list[dict[str, Any]]) -> CharacterBM25:
    return CharacterBM25([str(candidate.get("text_b", "")) for candidate in candidates])


def lexical_bm25_rank_indexed(
    query: str,
    bm25: CharacterBM25,
    candidates: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    scores = bm25.scores(query)
    ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[:top_k]
    max_score = max([score for _, score in ranked], default=0.0)
    rows: list[dict[str, Any]] = []
    for index, score in ranked:
        prob = float(score / max_score) if max_score > 0 else 0.0
        rows.append(dict(candidates[index], bm25_score=float(score), retrieval_score=float(score), prob_1=prob))
    return rows
