"""Small tests so that anyone cloning the repository can check it works.

Run with:  python -m pytest -q
"""
from gensim.models import Word2Vec

from src import compare, config, preprocess, train, viz


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


# ------------------------------------------------------------- web app helpers
def _toy_models():
    toy = [["student", "learn", "programming", "college", "course", "exam"]] * 40
    toy += [["river", "water", "flow", "stream", "lake", "rain"]] * 40
    params = dict(config.W2V_PARAMS, min_count=1, epochs=3, vector_size=20)
    cbow, skip, _ = train.train_models(toy, params)
    return cbow, skip


def test_training_examples_for_the_assignment_sentence():
    tokens = viz.demo_tokens("Students learn programming in college.")
    assert tokens == ["students", "learn", "programming", "in", "college"]
    cbow, skip = viz.training_examples(tokens, window=2)
    assert len(cbow) == 5                      # one example per centre word
    assert len(skip) == 14                     # 2 + 3 + 4 + 3 + 2 context words
    centre = [e for e in cbow if e["target"] == "programming"][0]
    assert centre["context"] == ["students", "learn", "in", "college"]


def test_compare_word_and_frame_shapes():
    cbow, skip = _toy_models()
    res = viz.compare_word(cbow, skip, "student", topn=3)
    assert len(res["cbow"]) == 3 and len(res["skipgram"]) == 3
    assert res["overlap"] == len(res["shared"]) <= 3
    assert set(res["only_cbow"]).isdisjoint(res["shared"])
    assert -1 <= res["agreement"] <= 1


def test_pca_map_has_target_and_neighbours():
    cbow, skip = _toy_models()
    df, explained = viz.pca_map(cbow, "student", n_near=5, topn=3, other_top=["learn"])
    assert len(df) == 6 and df.kind.iloc[0] == "Target word"
    assert (df.kind == "Top 3 neighbours").sum() == 3
    assert 0 < explained <= 1


def test_load_models_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert train.load_models() is None
