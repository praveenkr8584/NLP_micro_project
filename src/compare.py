"""Compare what CBOW and Skip-gram learned.

Three levels of evidence are produced:
1. the required table: top-5 neighbours of the five selected words
2. a numeric overlap score for those words and for the 200 most frequent words
3. extra checks: agreement of the two models' similarity scores, and a standard
   word-similarity test (WordSim-353, bundled with gensim)
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from scipy.stats import spearmanr

from . import config


# ------------------------------------------------------------------ 1. table
def neighbours(model: Word2Vec, word: str, topn: int = config.TOPN) -> List[tuple]:
    """[(neighbour, cosine_similarity), ...] for one word."""
    return model.wv.most_similar(word, topn=topn)


def comparison_table(
    cbow: Word2Vec, skip: Word2Vec, words: List[str], topn: int = config.TOPN
) -> pd.DataFrame:
    """The comparison table required by the assignment."""
    rows = []
    for word in words:
        c = [w for w, _ in neighbours(cbow, word, topn)]
        s = [w for w, _ in neighbours(skip, word, topn)]
        common = [w for w in c if w in s]
        rows.append(
            {
                "Word": word,
                "Frequency": cbow.wv.get_vecattr(word, "count"),
                "CBOW top 5": ", ".join(c),
                "Skip-gram top 5": ", ".join(s),
                "Common words": ", ".join(common) if common else "-",
                "Overlap": f"{len(common)}/{topn}",
            }
        )
    return pd.DataFrame(rows)


def neighbours_with_scores(
    cbow: Word2Vec, skip: Word2Vec, words: List[str], topn: int = config.TOPN
) -> pd.DataFrame:
    """Long table (one row per neighbour) with cosine scores - for the report."""
    rows = []
    for word in words:
        for model_name, model in (("CBOW", cbow), ("Skip-gram", skip)):
            for rank, (neighbour, score) in enumerate(neighbours(model, word, topn), 1):
                rows.append(
                    {
                        "Word": word,
                        "Model": model_name,
                        "Rank": rank,
                        "Neighbour": neighbour,
                        "Cosine": round(float(score), 3),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. overlap
def overlap_at_k(cbow: Word2Vec, skip: Word2Vec, word: str, k: int = config.TOPN) -> int:
    """How many of the top-k neighbours are the same in both models."""
    c = {w for w, _ in neighbours(cbow, word, k)}
    s = {w for w, _ in neighbours(skip, word, k)}
    return len(c & s)


def frequent_words(model: Word2Vec, n: int) -> List[str]:
    """The n most frequent words in the model vocabulary (gensim sorts by count)."""
    return list(model.wv.index_to_key[:n])


def overlap_summary(
    cbow: Word2Vec, skip: Word2Vec, n: int = config.N_FREQUENT_FOR_OVERLAP, k: int = config.TOPN
) -> Dict:
    """Average top-k overlap over the n most frequent words, with a chance level."""
    words = frequent_words(cbow, n)
    overlaps = np.array([overlap_at_k(cbow, skip, w, k) for w in words])
    vocab = len(cbow.wv)
    return {
        "words_checked": n,
        "k": k,
        "mean_overlap": round(float(overlaps.mean()), 2),
        "share_with_at_least_1_common": round(float((overlaps >= 1).mean()), 3),
        "share_with_3_or_more_common": round(float((overlaps >= 3).mean()), 3),
        "chance_level_mean_overlap": round(k * k / (vocab - 1), 4),
        "distribution": {int(v): int((overlaps == v).sum()) for v in range(k + 1)},
    }


# ------------------------------------------------------------------ 3. extras
def similarity_agreement(cbow: Word2Vec, skip: Word2Vec, word: str) -> float:
    """Spearman correlation of the two models' similarity scores for one word.

    Both models are asked: "how similar is `word` to EVERY other word?"
    A value near 1 means they rank the whole vocabulary in a similar order.
    """
    assert cbow.wv.index_to_key == skip.wv.index_to_key, "vocabularies differ"
    c = cbow.wv.cosine_similarities(cbow.wv[word], cbow.wv.vectors)
    s = skip.wv.cosine_similarities(skip.wv[word], skip.wv.vectors)
    mask = np.arange(len(c)) != cbow.wv.key_to_index[word]  # ignore the word itself
    rho, _ = spearmanr(c[mask], s[mask])
    return float(rho)


def neighbour_profile(
    model: Word2Vec, n: int = config.N_FREQUENT_FOR_OVERLAP, k: int = config.TOPN
) -> Dict:
    """What kind of neighbours does a model prefer?

    For the n most frequent words, look at their top-k neighbours and report
    how common those neighbours are in the corpus (median count) and how high
    the cosine scores are. Cosine values are only a rough guide when comparing
    two different models.
    """
    counts, scores = [], []
    for word in frequent_words(model, n):
        for neighbour, score in model.wv.most_similar(word, topn=k):
            counts.append(model.wv.get_vecattr(neighbour, "count"))
            scores.append(score)
    return {
        "median_neighbour_count_in_corpus": int(np.median(counts)),
        "mean_cosine_of_top5": round(float(np.mean(scores)), 3),
    }


def wordsim_evaluation(model: Word2Vec) -> Dict:
    """Standard test: do model similarities agree with human judgements?

    WordSim-353 has 353 word pairs rated by people. Pairs with a word that is
    not in our (small, stopword-free) vocabulary are skipped.
    """
    pearson, spearman, oov = model.wv.evaluate_word_pairs(
        datapath("wordsim353.tsv"), dummy4unknown=False
    )
    return {
        "spearman": round(float(spearman[0]), 3),
        "pearson": round(float(pearson[0]), 3),
        "pairs_not_covered_percent": round(float(oov), 1),
    }
