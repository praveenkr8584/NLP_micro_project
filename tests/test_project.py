"""Small tests so that anyone cloning the repository can check it works.

Run with:  python -m pytest -q
"""
from gensim.models import Word2Vec

from src import compare, config, preprocess, train


def test_clean_sentence_rules():
    words = ["The", "Students'", "learn", "programming", "in", "college", "2024", "."]
    # lowercase, letters only, >=3 letters, no stopwords
    assert preprocess.clean_sentence(words) == ["students", "learn", "programming", "college"]


def test_hyphen_is_split():
    assert preprocess.clean_sentence(["term-end"]) == ["term", "end"]


def test_shared_settings_do_not_contain_sg():
    assert "sg" not in config.W2V_PARAMS


def test_models_differ_only_in_sg():
    toy = [["student", "learn", "programming", "college", "course"]] * 50
    params = dict(config.W2V_PARAMS, min_count=1, epochs=2)
    cbow, skip, _ = train.train_models(toy, params)
    a, b = train.model_settings(cbow), train.model_settings(skip)
    assert a["sg"] == 0 and b["sg"] == 1
    a.pop("sg"), b.pop("sg")
    assert a == b


def test_overlap_is_between_zero_and_k():
    toy = [["a1x", "b2x", "c3x", "d4x", "e5x", "f6x"]] * 30
    model = Word2Vec(toy, sg=0, vector_size=10, min_count=1, epochs=2, workers=1, seed=1)
    other = Word2Vec(toy, sg=1, vector_size=10, min_count=1, epochs=2, workers=1, seed=1)
    assert 0 <= compare.overlap_at_k(model, other, "a1x", k=3) <= 3
