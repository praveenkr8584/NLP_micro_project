# Model Settings

**Library:** gensim 4.4.0 (`gensim.models.Word2Vec`)

Both models were trained on the **same cleaned corpus** with **identical settings**.
**Only the `sg` parameter is different.** The notebook prints both models' settings side by side
and asserts that `sg` is the only difference.

| Setting | CBOW | Skip-gram | Meaning |
|---|---|---|---|
| `sg` | **0** | **1** | 0 = CBOW (context predicts centre word), 1 = Skip-gram (centre word predicts context) |
| `vector_size` | 100 | 100 | length of every word vector |
| `window` | 5 | 5 | context words taken on each side of the centre word |
| `min_count` | 5 | 5 | words with fewer than 5 occurrences are ignored |
| `epochs` | 20 | 20 | number of passes over the corpus |
| `negative` | 5 | 5 | negative samples per training example (negative sampling, `hs=0`) |
| `sample` | 0.001 | 0.001 | random down-sampling of very frequent words |
| `workers` | 1 | 1 | one thread, so the result does not depend on thread timing |
| `seed` | 42 | 42 | fixed random seed |
| vocabulary size | 12,982 | 12,982 | words that survived `min_count` (identical, as expected) |

Other gensim settings keep their default values: initial learning rate `alpha=0.025`, final learning rate `min_alpha=0.0001`. Both are now recorded in `results/model_settings.json` on every run.

## Why these values
* **vector_size=100, window=5, min_count=5** are common defaults and a good fit for a corpus of this size.
* **epochs=20** (gensim's default is 5) because the corpus is small and needs more passes.
* **workers=1 and seed=42** make the experiment repeatable. The full pipeline was run three times and
  gave exactly the same neighbours and scores each time.

## Training time (on the machine used to build the project)
CBOW 29.8 seconds, Skip-gram 81.9 seconds (~2.7× slower). Skip-gram is slower because it creates one
training example for every context word, while CBOW creates one example per centre word.

## Similarity measure
`model.wv.most_similar(word, topn=5)` returns the words whose vectors have the highest
**cosine similarity** with the vector of the given word.

## The five selected words
`school`, `government`, `water`, `war`, `house`

Selection rule: nouns from different topic areas that are inside the 100 most frequent words of the cleaned corpus
(ranks 38, 51, 49, 40 and 29). This tests agreement across subjects, not only in one topic.
