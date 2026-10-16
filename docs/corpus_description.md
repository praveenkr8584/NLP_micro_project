# Corpus Description

**Project:** Word Similarity Using Word2Vec Models (CBOW vs Skip-gram)
**Name:** Praveen Kumar | **Reg. No.:** 231FA04G42

## Source
The **Brown Corpus** (W. N. Francis and H. Kucera, Brown University, 1964; revised 1971 and 1979),
a standard collection of present-day edited American English.
The corpus README states that redistribution is permitted, so the original file is stored in
`data/raw/brown.zip`. It was taken from the NLTK data repository on GitHub
(`nltk/nltk_data`, `packages/corpora/brown.zip`).

It contains **500 text samples** (about 2,000 words each) from **15 genres**:
news reportage, editorials, reviews, religion, skills and hobbies, popular lore, belles-lettres,
government documents, learned (scientific) writing, general fiction, mystery, science fiction,
adventure, romance and humour. This mixture of topics is the reason the corpus is useful
for word similarity: the same word appears in many different contexts.

## Why this corpus
* The assignment asks for **several thousand sentences**; Brown has more than 57,000.
* It is clean, well known and free to redistribute, so the project is repeatable by anyone.
* It needs no manual download or login.

## Size (measured by `python main.py`)

| Measure | Value |
|---|---|
| Text samples (files) | 500 |
| Sentences in the original corpus (one per line) | 57,340 |
| Words in the original corpus (including punctuation tokens) | 1,161,192 |
| **Sentences kept after cleaning** | **44,509** |
| **Tokens after cleaning** | **448,314** |
| Vocabulary (different words) after cleaning | 40,209 |
| Words that appear at least 5 times (the model vocabulary) | 12,982 |
| Average sentence length after cleaning | 10.07 tokens |

Sentences kept per genre letter: a 3,898 · b 2,398 · c 1,516 · d 1,373 · e 3,523 · f 4,261 ·
g 6,265 · h 2,427 · j 6,835 · k 2,681 · l 2,339 · m 568 · n 2,942 · p 2,696 · r 787.

## Format of the original data
Every line is one sentence. Every word carries a part-of-speech tag, for example
`The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd ...`.
Only the word is used; the tag is removed.

## Cleaning and tokenizing (`src/preprocess.py`)
1. Split every token at `/` and keep the word.
2. Convert to lowercase.
3. Keep letters only: numbers and punctuation are removed, and hyphens/apostrophes split a word
   (`term-end` becomes `term`, `end`).
4. Remove words shorter than 3 letters.
5. Remove English stopwords (the scikit-learn list of 318 words, e.g. *the, of, and, was*).
6. Remove sentences with fewer than 4 remaining words.

Example:

| Step | Sentence |
|---|---|
| Original | The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced `` no evidence '' that any irregularities took place . |
| Cleaned | fulton county grand jury said friday investigation atlanta recent primary election produced evidence irregularities took place |

The cleaned corpus is saved in `data/processed/brown_sentences.txt` (one sentence per line,
words separated by spaces), which is exactly what Word2Vec receives.

## Limitations of the corpus
* About 448,000 tokens is small for Word2Vec (it was designed for billions), so vectors for rarer words are noisy.
* The text is from 1961, so some words reflect that time (for example, place names and the topics in the news).
* Removing stopwords makes the words that stay closer to each other inside the window.
* Lines of the original corpus are not always full sentences (a few are fragments).
