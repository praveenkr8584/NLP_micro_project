"""Central settings for the project.

Everything that must stay identical between the CBOW and Skip-gram runs lives
here, so there is exactly one place to read (and to explain in a viva).
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "brown.zip"
PROCESSED_TXT = ROOT / "data" / "processed" / "brown_sentences.txt"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"  # git-ignored (models are re-created in ~1 minute)

# ---------------------------------------------------------------- corpus
# Brown Corpus (Francis & Kucera). Redistribution is permitted by its README.
BROWN_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
    "packages/corpora/brown.zip"
)
MIN_TOKEN_LEN = 3      # drop words shorter than 3 letters
MIN_SENTENCE_LEN = 4   # drop sentences with fewer than 4 remaining words

# ---------------------------------------------------------------- Word2Vec
# These settings are SHARED by both models. The ONLY difference is `sg`.
SEED = 42
W2V_PARAMS = dict(
    vector_size=100,   # length of every word vector
    window=5,          # context words taken on each side of the centre word
    min_count=5,       # ignore words that appear fewer than 5 times
    epochs=20,         # passes over the corpus (small corpus -> more passes)
    negative=5,        # negative samples per training example
    sample=1e-3,       # down-sampling of very frequent words
    workers=1,         # 1 worker + fixed seed = repeatable results
    seed=SEED,
)

# ---------------------------------------------------------------- analysis
TOPN = 5               # neighbours compared for each selected word
# Five frequent nouns from DIFFERENT topic areas, all inside the 100 most
# frequent words of the cleaned corpus: education, politics, nature, conflict
# and everyday life ("house" is also ambiguous, which makes it interesting).
SELECTED_WORDS = ["school", "government", "water", "war", "house"]
N_FREQUENT_FOR_OVERLAP = 200  # extra check over the 200 most frequent words
