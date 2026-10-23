"""Train the CBOW and Skip-gram models with identical settings."""
import time
import warnings
from typing import Dict, List, Optional, Tuple

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
        "alpha": model.alpha,
        "min_alpha": model.min_alpha,
        "workers": model.workers,
        "seed": model.seed,
        "vocabulary": len(model.wv),
    }


def save_models(cbow: Word2Vec, skip: Word2Vec) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cbow.save(str(config.MODELS_DIR / "cbow.model"))
    skip.save(str(config.MODELS_DIR / "skipgram.model"))


def load_models() -> Optional[Tuple[Word2Vec, Word2Vec]]:
    """Load the saved models, or return None if they are missing or out of date.

    "Out of date" means the saved settings differ from config.W2V_PARAMS, so the
    web app never shows results that do not match the current settings.
    """
    cbow_path = config.MODELS_DIR / "cbow.model"
    skip_path = config.MODELS_DIR / "skipgram.model"
    if not (cbow_path.exists() and skip_path.exists()):
        return None
    try:
        cbow = Word2Vec.load(str(cbow_path))
        skip = Word2Vec.load(str(skip_path))
    except Exception as exc:  # damaged file or incompatible gensim version
        warnings.warn(f"Could not load saved models ({exc}); they will be retrained.")
        return None
    checked = ("vector_size", "window", "min_count", "epochs", "negative", "sample", "seed")
    for model, sg in ((cbow, 0), (skip, 1)):
        if model.sg != sg:
            return None
        if any(getattr(model, key) != config.W2V_PARAMS[key] for key in checked):
            return None
    return cbow, skip
