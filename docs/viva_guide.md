# Viva Guide

Short answers in simple English. Say them in your own words; practise them aloud.
Tense tip: describe **what you did** in the **past simple** ("I trained", "I removed", "I chose"),
and describe **facts and how things work** in the **present simple** ("CBOW predicts", "Skip-gram uses").

## 1. Introduce the project (30 seconds)
"My project is called *Word Similarity Using Word2Vec Models*. I **trained** two models, CBOW and Skip-gram,
on the same corpus, the Brown Corpus. I **kept** all settings the same, so the comparison was fair.
Then I **selected** five frequent words and **compared** their five most similar words in both models.
I **found** that the models learn related relationships, but their top-five lists are mostly different."

## 2. Questions about the concepts

**What is Word2Vec?**
It is a method that turns every word into a vector of numbers. Words that appear in similar contexts get similar vectors.

**What is the difference between CBOW and Skip-gram?**
CBOW uses the surrounding words to predict the centre word. Skip-gram uses the centre word to predict the surrounding words.
In my example, CBOW uses *students, learn, in, college* to predict *programming*; Skip-gram uses *programming* to predict each of them.

**Which one is faster? Why?**
CBOW is faster. It makes one training example per centre word. Skip-gram makes one example for every context word.
In my project CBOW took about 13 seconds and Skip-gram about 38 seconds.

**Which one is usually better for rare words?**
Skip-gram. Each context word gets its own training example, so rare words get more updates.
In my results, Skip-gram preferred rare words as neighbours, but some of them were noisy.

**What does `sg` mean?**
It is a gensim parameter. `sg=0` is CBOW and `sg=1` is Skip-gram.

**What do window, vector_size, min_count and epochs mean?**
`window=5`: five words on each side are context. `vector_size=100`: each word has 100 numbers.
`min_count=5`: words with fewer than five occurrences are ignored. `epochs=20`: the model reads the corpus 20 times.

**What is negative sampling?**
For every real (word, context) pair, the model also sees a few random wrong pairs (I used 5). This makes training fast.

**How is similarity measured?**
By cosine similarity between two word vectors. It measures the angle between them: 1 means the same direction.

## 3. Questions about my project

**Why did you choose the Brown Corpus?**
It has more than 57,000 sentences from 15 genres, it is clean, it is free to redistribute and needs no manual download.

**How did you clean the text?**
I converted words to lowercase, kept letters only, removed words shorter than three letters, removed stopwords,
and removed sentences with fewer than four words.

**Why did you remove stopwords?**
Words like *the* and *of* are everywhere and carry little meaning, so they would crowd the neighbour lists.
The disadvantage is that it changes which words are next to each other in the window.

**How many sentences and words did you use?**
44,509 sentences and 448,314 tokens after cleaning. The vocabulary was 12,982 words with at least 5 occurrences.

**Why did you keep the settings the same?**
To make the comparison fair. If more than one setting changes, I cannot know what caused the difference.
My notebook prints both settings and checks that only `sg` is different.

**How did you choose the five words?**
I took frequent nouns from different topics inside the 100 most frequent words: school, government, water, war, house.

**What were your results?**
Only 2 of 25 top-five slots were shared (*governments* and *civil*). Over the 200 most frequent words the average overlap
was 0.63 out of 5. But the similarity scores over the whole vocabulary correlated between 0.72 and 0.81.
So the models agree on the broad picture and disagree on the exact top five.

**Which model was better?**
It depends on the goal. CBOW gave frequent, general neighbours (*schools, teacher, ice, salt*). Skip-gram gave rarer,
more specific neighbours (*desegregation, reservoir, propaganda*) and agreed better with human ratings on WordSim-353
(0.446 against 0.296).

**Why are some neighbours strange (for example *sox* for *house*, *balafrej* for *government*)?**
The corpus is small. A word that appears only a few times can be close to another word because of a few sentences.
*House* also has several meanings, which mix into one vector.

**How do you know your results are repeatable?**
I used `workers=1` and `seed=42`, and I ran the pipeline several times; the neighbours and scores were identical.

**What are the limitations?**
Small corpus, one seed, one set of settings, stopwords removed, and judging "meaningful" is partly subjective.

**What would you do next?**
Visualise the vectors in 2D with PCA or t-SNE, test several seeds and window sizes, and use a bigger corpus.

## 4. Useful sentences to practise aloud
* "I trained two models on the same corpus."
* "CBOW predicts the target word from its context."
* "Skip-gram predicts the context words from the target word."
* "I kept the vector size, window and training settings the same."
* "The two models gave different top-five lists, but their overall similarity scores agree."
* "Skip-gram's neighbours were rarer and more specific."
* "A larger corpus would make the results more reliable."
