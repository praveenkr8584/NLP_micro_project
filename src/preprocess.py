"""Clean and tokenize the corpus.

Cleaning steps (all of them are listed again in docs/corpus_description.md):
1. lowercase every word
2. keep letters only (numbers, punctuation, hyphens and apostrophes split words)
3. remove words shorter than 3 letters
4. remove English stopwords (scikit-learn list)
5. remove sentences that have fewer than 4 words left
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import config
from .data import iter_raw_sentences

_NON_LETTERS = re.compile(r"[^a-z]+")


def clean_sentence(
    words: List[str],
    min_len: int = config.MIN_TOKEN_LEN,
    stopwords=ENGLISH_STOP_WORDS,
) -> List[str]:
    """Turn the raw words of one sentence into clean tokens."""
    tokens = []
    for word in words:
        # "term-end" -> "term", "end"; "Atlanta's" -> "atlanta", "s"
        for piece in _NON_LETTERS.split(word.lower()):
            if len(piece) >= min_len and piece not in stopwords:
                tokens.append(piece)
    return tokens


def build_corpus(min_sentence_len: int = config.MIN_SENTENCE_LEN):
    """Read Brown and return (sentences, stats).

    sentences: list of token lists, e.g. [["fulton", "county", "grand", ...], ...]
    stats:     numbers needed for the corpus description
    """
    sentences: List[List[str]] = []
    files = set()
    raw_lines = 0
    raw_tokens = 0
    genre_sentences: Counter = Counter()

    for file_name, words in iter_raw_sentences():
        files.add(file_name)
        raw_lines += 1
        raw_tokens += len(words)
        tokens = clean_sentence(words)
        if len(tokens) >= min_sentence_len:
            sentences.append(tokens)
            genre_sentences[file_name[1]] += 1

    counts = Counter(t for s in sentences for t in s)
    n_tokens = sum(counts.values())
    stats = {
        "source": "Brown Corpus (Francis & Kucera, 1964/1979)",
        "text_samples": len(files),
        "raw_sentences": raw_lines,
        "raw_tokens_incl_punctuation": raw_tokens,
        "sentences_kept": len(sentences),
        "tokens_after_cleaning": n_tokens,
        "vocabulary_size": len(counts),
        "vocabulary_min_count_5": sum(1 for c in counts.values() if c >= 5),
        "avg_sentence_length": round(n_tokens / len(sentences), 2),
        "sentences_per_genre_letter": dict(sorted(genre_sentences.items())),
    }
    return sentences, stats


def save_sentences(sentences: List[List[str]], path: Path = config.PROCESSED_TXT) -> Path:
    """Save one sentence per line (tokens separated by spaces)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(s) for s in sentences) + "\n", encoding="utf-8")
    return path


def load_sentences(path: Path = config.PROCESSED_TXT) -> List[List[str]]:
    """Load the file written by `save_sentences`."""
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_stats(stats: Dict, path: Path = config.RESULTS_DIR / "corpus_stats.json") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path
