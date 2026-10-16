# Results and Explanation

**Question:** Do CBOW and Skip-gram learn similar relationships between words?

**Short answer:** They learn **related but not identical** relationships. They agree on the broad
structure of the vocabulary, but they often disagree about which five words are the closest.

All numbers come from `python main.py` / the notebook and are saved in `results/`.

## 1. Comparison table (top 5 similar words)

| Word | Frequency | CBOW top 5 | Skip-gram top 5 | Common | Overlap |
|---|---|---|---|---|---|
| school | 502 | schools, teacher, divinity, graduate, structured | retarded, desegregation, elementary, graduates, mentally | – | 0/5 |
| government | 432 | constitution, foreign, governments, coalition, abolish | belgian, governments, balafrej, currency, yugoslavia | governments | 1/5 |
| water | 442 | ice, hot, salt, cooled, soil | rinse, bucket, frontage, boil, reservoir | – | 0/5 |
| war | 482 | renowned, battle, citizen, civil, wars | civil, fronts, korean, custer, propaganda | civil | 1/5 |
| house | 573 | sox, hotel, mob, kent, apartment | siding, coffin, barber, livery, fireplace | – | 0/5 |

The cosine scores are in `results/neighbours_with_scores.csv`.

## 2. Which relationships look meaningful?

| Word | Meaningful neighbours | Weak or noisy neighbours |
|---|---|---|
| school | CBOW: *schools, teacher, graduate*. Skip-gram: *elementary, graduates, desegregation* | CBOW: *structured, divinity* |
| government | CBOW: *constitution, governments, coalition, foreign*. Skip-gram: *governments, currency, yugoslavia, belgian* | Skip-gram: *balafrej* (a proper name that is rare in the corpus) |
| water | CBOW: *ice, hot, salt, cooled, soil*. Skip-gram: *rinse, boil, bucket, reservoir* | Skip-gram: *frontage* |
| war | CBOW: *battle, wars, civil*. Skip-gram: *civil, korean, propaganda, fronts* | CBOW: *renowned*; Skip-gram: *custer* |
| house | CBOW: *hotel, apartment*. Skip-gram: *fireplace, coffin, barber, livery, siding* | CBOW: *sox, kent, mob* ("house" has several meanings) |

Judgement: "meaningful" means a person would accept the word as related (same topic, part of the same
thing, or an inflected form). This is a human judgement, so it is subjective.

## 3. Numbers behind the answer

| Evidence | CBOW | Skip-gram |
|---|---|---|
| Shared top-5 neighbours, five selected words | 2 of 25 slots | |
| Mean top-5 overlap, 200 most frequent words | 0.63 of 5 (chance level: 0.002) | |
| Share of those 200 words with at least 1 common neighbour | 41% | |
| Share with 3 or more common neighbours | 4.5% | |
| Spearman agreement of full-vocabulary similarity, selected words | 0.72 to 0.81 | |
| Median corpus count of the neighbours | 52 | 9 |
| Mean cosine score of the top 5 | 0.718 | 0.521 |
| WordSim-353 Spearman (75% of the pairs covered) | 0.296 | 0.446 |

## 4. Explanation

1. **The top-5 lists are mostly different.** Only *governments* and *civil* appear in both lists.
   Over the 200 most frequent words, the average overlap is 0.63 out of 5. Random lists would overlap
   about 0.002, so the models clearly share information, but the exact top five rarely match.
2. **The wider picture is similar.** When both models score *every* word against a given word, their
   orderings correlate between 0.72 and 0.81. So both place the same kinds of words near and far;
   the difference is mostly which close words come first.
3. **CBOW prefers common, general neighbours.** Its neighbours have a median corpus count of 52 and higher
   cosine scores. Examples: *schools, teacher, constitution, governments, ice, salt, battle*.
4. **Skip-gram prefers rare, specific neighbours.** The median count is 9. This gives topic-specific words
   (*desegregation, reservoir, propaganda, fireplace*), but also words seen only a few times, such as the
   name *balafrej*, where the vector is less reliable.
5. **Skip-gram agrees better with human similarity ratings** on WordSim-353 (0.446 against 0.296), measured
   on the pairs our vocabulary covers.

## 5. Conclusion
CBOW and Skip-gram both learn meaningful topical relations (school – elementary/graduate,
war – battle/civil, water – boil/ice), so they learn **similar kinds of relationships**. They do **not** give
the same closest words. CBOW favours frequent, general neighbours; Skip-gram favours rarer, more specific
neighbours and agreed better with human ratings in this experiment.

## 6. Limitations
* Small corpus (448,314 tokens): neighbours of less frequent words are noisy.
* One corpus, one seed and one set of settings. Other values of `window` or `vector_size` could change the lists.
* Stopwords were removed, which changes which words are "next to" each other.
* Words with several meanings (*house*) mix their contexts into one vector.
* Judging "meaningful" is subjective; WordSim-353 gives an independent check but covers only 75% of the pairs.

## 7. Possible extensions
Visualise the five words and their neighbours in 2D (PCA or t-SNE), repeat with several seeds, vary
`window`, or train on a larger corpus.
