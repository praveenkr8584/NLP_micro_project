# Word Similarity Using Word2Vec Models: CBOW vs Skip-gram

Micro project (22.10.26 to 25.10.26) · **Praveen Kumar** · Reg. No. **231FA04G42**

**Question:** Do CBOW and Skip-gram learn similar relationships between words?

I trained two Word2Vec models (CBOW and Skip-gram) on the same corpus, with identical settings,
and compared the five most similar words for five frequent words.

## Result in one minute

| Word | CBOW top 5 | Skip-gram top 5 | Common |
|---|---|---|---|
| school | schools, teacher, divinity, graduate, structured | retarded, desegregation, elementary, graduates, mentally | – |
| government | constitution, foreign, governments, coalition, abolish | belgian, governments, balafrej, currency, yugoslavia | governments |
| water | ice, hot, salt, cooled, soil | rinse, bucket, frontage, boil, reservoir | – |
| war | renowned, battle, citizen, civil, wars | civil, fronts, korean, custer, propaganda | civil |
| house | sox, hotel, mob, kent, apartment | siding, coffin, barber, livery, fireplace | – |

* The models learn **related but not identical** relationships: only 2 of 25 top-5 slots are shared, and the
  average overlap over the 200 most frequent words is 0.63 of 5 (chance level 0.002).
* Their similarity scores over the whole vocabulary agree fairly well (Spearman 0.72 to 0.81).
* **CBOW** prefers frequent, general neighbours; **Skip-gram** prefers rarer, more specific neighbours and agreed
  better with human ratings on WordSim-353 (0.446 vs 0.296).

Full explanation: [`docs/results_explanation.md`](docs/results_explanation.md)

## What is submitted

| Required item | Where |
|---|---|
| Python notebook | [`notebooks/word2vec_comparison.ipynb`](notebooks/word2vec_comparison.ipynb) (outputs included) |
| Corpus description | [`docs/corpus_description.md`](docs/corpus_description.md) |
| Model settings | [`docs/model_settings.md`](docs/model_settings.md) |
| Comparison table | in the notebook, and [`results/comparison_table.csv`](results/comparison_table.csv) |
| Short explanation of the results | notebook section 8, and [`docs/results_explanation.md`](docs/results_explanation.md) |

Also included: [`docs/viva_guide.md`](docs/viva_guide.md) (questions and answers for the viva).

## Project structure

```
word2vec-cbow-vs-skipgram/
├── README.md
├── main.py                      run the whole pipeline from the command line
├── requirements.txt             libraries (flexible versions)
├── requirements-lock.txt        exact versions the project was tested with
├── data/
│   ├── raw/brown.zip            Brown Corpus (redistribution permitted)
│   └── processed/brown_sentences.txt   cleaned sentences, one per line
├── src/
│   ├── config.py                all settings in one place
│   ├── data.py                  read the Brown Corpus
│   ├── preprocess.py            clean and tokenize
│   ├── train.py                 train CBOW and Skip-gram (only sg differs)
│   └── compare.py               tables, overlap, agreement, WordSim test
├── notebooks/word2vec_comparison.ipynb
├── docs/                        corpus description, model settings, results, viva guide
├── results/                     CSV and JSON files produced by the run
└── tests/test_project.py        5 small tests
```

## How to run

**Requirements:** Python 3.10 or newer (the project was tested with Python 3.13).

```bash
# 1. create a virtual environment (nothing is installed globally)
python -m venv .venv

# 2. activate it
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS

# 3. install the libraries
pip install -r requirements.txt

# 4a. run the notebook
jupyter notebook notebooks/word2vec_comparison.ipynb

# 4b. or run everything from the command line (about 1 minute)
python main.py

# 5. optional: run the tests
python -m pytest -q
```

`python main.py` rewrites the files in `results/`. No internet is needed because the corpus is in the repository.

## Method in short

1. **Corpus:** Brown Corpus, 500 texts of 15 genres → 44,509 sentences, 448,314 tokens after cleaning.
2. **Cleaning:** lowercase, letters only, words of at least 3 letters, stopwords removed, sentences of at least 4 words.
3. **Models:** gensim `Word2Vec`, `vector_size=100, window=5, min_count=5, epochs=20, negative=5, seed=42, workers=1`;
   `sg=0` (CBOW) and `sg=1` (Skip-gram). The notebook proves that `sg` is the only difference.
4. **Words:** school, government, water, war, house (frequent nouns from different topics).
5. **Evidence:** top-5 table, overlap over 200 frequent words, whole-vocabulary agreement, neighbour frequency,
   WordSim-353 test.

## Reproducibility
Workers = 1 and a fixed seed make the results repeatable: the pipeline gave identical neighbours and scores
in every run on the machine used to build it (gensim 4.4.0, numpy 2.5.4). Another gensim or numpy version may
change some scores slightly. Exact versions are in `requirements-lock.txt`.

## Limitations
Small corpus, one seed, one set of settings, stopwords removed, and ambiguous words such as *house*.
See `docs/results_explanation.md`.

## Possible next step
A 2D visualisation (PCA or t-SNE) of the five words and their neighbours.

## Data licence and credits
* Brown Corpus: W. N. Francis and H. Kucera, Brown University (1964, revised 1971 and 1979). Its README allows
  redistribution. The zip was taken from the NLTK data repository (`nltk/nltk_data`).
* Word2Vec: Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*.
* Library: gensim (Rehurek and Sojka). WordSim-353 file bundled with gensim (Finkelstein et al., 2002).
