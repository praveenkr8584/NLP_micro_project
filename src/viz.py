"""Data helpers for the web app (app.py).

There is no Streamlit code in this file, so every function can be tested and
reused in the notebook.
"""
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from . import compare, config


# ------------------------------------------------------------ explore one word
def neighbour_frame(
    model: Word2Vec, word: str, topn: int, other_words: Sequence[str] = ()
) -> pd.DataFrame:
    """Top-n neighbours of `word` with cosine score, corpus count and a 'shared' flag."""
    other = set(other_words)
    rows = []
    for rank, (neighbour, score) in enumerate(model.wv.most_similar(word, topn=topn), 1):
        rows.append(
            {
                "rank": rank,
                "word": neighbour,
                "cosine": round(float(score), 3),
                "count": int(model.wv.get_vecattr(neighbour, "count")),
                "shared": neighbour in other,
            }
        )
    return pd.DataFrame(rows)


def compare_word(cbow: Word2Vec, skip: Word2Vec, word: str, topn: int) -> Dict:
    """Everything the 'Explore a word' tab needs for one word."""
    c_words = [w for w, _ in cbow.wv.most_similar(word, topn=topn)]
    s_words = [w for w, _ in skip.wv.most_similar(word, topn=topn)]
    shared = [w for w in c_words if w in s_words]
    return {
        "cbow": neighbour_frame(cbow, word, topn, s_words),
        "skipgram": neighbour_frame(skip, word, topn, c_words),
        "shared": shared,
        "overlap": len(shared),
        "only_cbow": [w for w in c_words if w not in s_words],
        "only_skipgram": [w for w in s_words if w not in c_words],
        "agreement": compare.similarity_agreement(cbow, skip, word),
        "count": int(cbow.wv.get_vecattr(word, "count")),
        "rank": cbow.wv.key_to_index[word] + 1,
    }


# ------------------------------------------------------------------- 2D word map
def pca_map(
    model: Word2Vec, word: str, n_near: int, topn: int, other_top: Sequence[str]
) -> Tuple[pd.DataFrame, float]:
    """2D PCA of `word` and its n_near nearest neighbours in ONE model.

    Returns (table, share of variance explained by the 2 axes). Positions can only be
    compared inside one model: the two models have different vector spaces.
    """
    near = model.wv.most_similar(word, topn=n_near)
    words = [word] + [w for w, _ in near]
    cosines = [1.0] + [float(s) for _, s in near]
    vectors = np.array([model.wv[w] for w in words])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    pca = PCA(n_components=2, random_state=config.SEED)
    xy = pca.fit_transform(vectors)

    other = set(other_top)
    target_label = "Target word"
    top_label = f"Top {topn} neighbours"
    far_label = "Other nearby words"
    kinds = [target_label] + [top_label if i < topn else far_label for i in range(n_near)]
    df = pd.DataFrame(
        {
            "word": words,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cosine": [round(c, 3) for c in cosines],
            "count": [int(model.wv.get_vecattr(w, "count")) for w in words],
            "kind": kinds,
            "in_other_top": [(w in other) and i > 0 for i, w in enumerate(words)],
        }
    )
    df["label"] = [w + (" ◆" if shared else "") for w, shared in zip(df.word, df.in_other_top)]
    return df, float(pca.explained_variance_ratio_.sum())


# ----------------------------------------------------- "How it works" demo tab
def demo_tokens(sentence: str) -> List[str]:
    """Lowercase letters-only tokens (no stopword removal, so the demo stays readable)."""
    return re.findall(r"[a-z]+", sentence.lower())


def training_examples(tokens: List[str], window: int) -> Tuple[List[Dict], List[Dict]]:
    """Training examples that CBOW and Skip-gram would create from one sentence.

    CBOW:      one example per centre word  -> (all context words) -> centre word
    Skip-gram: one example per context word -> centre word -> one context word
    (gensim also shrinks the window randomly at each step, so `window` is the maximum.)
    """
    cbow_examples, skip_examples = [], []
    for i, target in enumerate(tokens):
        lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
        context = [tokens[j] for j in range(lo, hi) if j != i]
        if not context:
            continue
        cbow_examples.append({"centre_index": i, "target": target, "context": context})
        for c in context:
            skip_examples.append({"centre_index": i, "target": target, "context_word": c})
    return cbow_examples, skip_examples
