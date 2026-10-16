"""Train the CBOW and Skip-gram models with identical settings."""
import time
from typing import Dict, List, Tuple

from gensim.models import Word2Vec

from . import config


def train_models(
    sentences: List[List[str]], params: Dict = None
) -> Tuple[Word2Vec, Word2Vec, Dict[str, float]]:
    """Return (cbow, skipgram, seconds_taken).

    The only difference between the two calls is ``sg``:
      sg=0 -> CBOW      (context words predict the centre word)
      sg=1 -> Skip-gram (centre word predicts the context words)
    """
    params = dict(config.W2V_PARAMS if params is None else params)
    assert "sg" not in params, "sg must not be inside the shared settings"

    timings = {}

    start = time.perf_counter()
    cbow = Word2Vec(sentences, sg=0, **params)
    timings["cbow_seconds"] = round(time.perf_counter() - start, 1)

    start = time.perf_counter()
    skip = Word2Vec(sentences, sg=1, **params)
    timings["skipgram_seconds"] = round(time.perf_counter() - start, 1)

    return cbow, skip, timings


def model_settings(model: Word2Vec) -> Dict:
    """Read the settings back from a trained model (proof that they match)."""
    return {
        "sg": model.sg,
        "vector_size": model.vector_size,
        "window": model.window,
        "min_count": model.min_count,
        "epochs": model.epochs,
        "negative": model.negative,
        "sample": model.sample,
        "workers": model.workers,
        "seed": model.seed,
        "vocabulary": len(model.wv),
    }


def save_models(cbow: Word2Vec, skip: Word2Vec) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cbow.save(str(config.MODELS_DIR / "cbow.model"))
    skip.save(str(config.MODELS_DIR / "skipgram.model"))
