# Transformers.

Notes on Transformers taken while going through Jay Alammar's [blog posts](http://jalammar.github.io), Sebastian Ruder's [blog posts](https://www.ruder.io), and the HuggingFace NLP [course](https://huggingface.co/course).

## A Review of the Neural History of Natural Language Processing (upto ~2017)
Blog post from Sebastian Ruder on the history of neural NLP: [link](https://www.ruder.io/a-review-of-the-recent-history-of-nlp/).

## Word Embeddings
This section summarizes major word-embedding approaches upto the advent of Transformers.

`TODO`:
1. On word embeddings - Part 1 [sebastian-ruder-blog](https://www.ruder.io/word-embeddings-1).
2. On word embeddings - Part 2: Approximating the Softmax [sebastian-ruder-blog](https://www.ruder.io/word-embeddings-softmax/).
3. On word embeddings - Part 3: The secret ingredients of word2vec [sebastian-ruder-blog](https://www.ruder.io/secret-word2vec/).
4. Word embeddings in 2017: Trends and future directions [sebastian-ruder-blog](https://www.ruder.io/word-embeddings-2017/).

**One-Hot Encoding**: Simplest form of word $\rightarrow$ vector model is one-hot encoding, though it results in sparse vector representations. $\text{Word2Vec}$ results in dense vector representations.

**TF-IDF**:
* Before $\text{Word2Vec}$, apart from one-hot encoding, TF-IDF was one of the the most common way to represent words:    
$\text{tf-idf}(\text{term}, \text{document})=\text{tf}(\text{term},\text{document})\cdot\text{idf}(\text{term})$
  * $\text{tf}(\text{term}, \text{document})=\frac{n_i}{\sum_{k=1}^V n_k}$    
where, $n_i$ represents the frequency of term $i$ in the document, and $V$ is the vocabulary size.
  * $\text{idf}(\text{term})=\log{\frac{N}{n_{t_i}}}$    
where, $N$ represents the total number of documents and $n_{t_i}$ represents the number of documents that contain the term $i$.

### Word2Vec
First neural embedding model for learning distributed representations.    
Each word is represented with a fixed-size float embedding vector.

Three main ideas: 
1. **Continuous Bag of Words (CBOW)**: Predict a _word_ given the _context_.    
   Two types of context:
   1. Single-word Context
   2. Multi-word Context
2. **Skip-gram**: Predict the _context_ given a _word_.
3. **Negative Sampling** Instead of training the classifier on all $+$ word pairs: $(\text{word}_i, \text{word}_{i+1})$ derived in CBOW or in Skipgrams, train the classifier on word pairs: $(\text{word}, \text{random})$ as well.

**Speeding up training**: Projecting onto the entire vocabulary to select the predicted word is an expensive operation. Instead, train a logistic regression classifier for pairs of words: nearby word pairs are positives while far away word pairs are negatives.

#### Continuous Bag of Words
Single-word Context: Given a word, predict a word: $p(w_i|w_{i-1})$.    
Multi-word Context: Given several surrounding words, predict a word: $p(w_i|w_{i-2}, w_{i-1}, w_{i+1},w_{i+2})$.    

#### Skip-grams
Flip the Multi-word Context CBOW model.    
Given a word, predict the surrounding words: $p(w_{i-2}|w_i)$, $p(w_{i-1}|w_i)$, $p(w_{i+1}|w_i)$, $p(w_{i+2}|w_i)$.    


References:
1. The Illustrated Word2Vec: [link](https://jalammar.github.io/illustrated-word2vec/). Jay Alammar.
2. On Word Embeddings: [part1](https://www.ruder.io/word-embeddings-1), [part2](https://www.ruder.io/word-embeddings-softmax/), [part3](https://www.ruder.io/secret-word2vec/). Sebastian Ruder.


## Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)

### Neural Machine Translation Models
Seq2Seq models generally consist of and `encoder` and a `decoder`.    
The encoder processes each item in the input sequence and compiles the information it captures into a vector (called the `context`).    
After processing the entire input sequence, the encoder sends the context over to the decoder, which begins producing the output sequence item by item.

The encoders and decoders are generally recurrent neural nets - `RNNs` - such as LSTMs or GRUs.    
Input: vectorized representation of a `word` (say, a Word2Vec embedding) and a `hidden state` at each timestep.    
Output: vectorized `output` and the `hidden state` for the next timestep.    
The hidden state of the encoder at its last timestep is the context vector, which is passed to the decoder.    
The decoder also maintains a hidden state that it passes from one time step to the next.

### Attention
The `context` vector turned out to be a bottleneck for NMTs making it difficult to process long sentences.    
Instead of sending only the `context` into the `decoder`, the  **Attention** mechanism is able to focus on all relevant parts of the input sequence. Attention allows the `decoder to look at the `context` output by the `encoder` at different timesteps.

Major upgrades from NMTs:
1. The `encoder` passes a lot more data to the `decoder` (passing all `hidden states`, instead of only the last one).
2. At each timestep, the `decoder` parses the `hidden states` of the `encoder` and multiplies each state by a softmax-ed `score` for the timestep, thus amplifying the `hidden states` with high scores, and drowning out `hidden states` with low scores.

Decoder:
1. At each timestep, the `decoder` calculates a `score` for each `hidden state` of the `encoder` and then multiplies these `hidden state`s by its `score`.
2. This results in a `context` vector which has amplified relevant information for this particular timestep from all the `hidden state`s of the `encoder`.
3. This `context` is concatenated with the `hidden state` of the `decoder` and used as input to a jointly trained `MLP` that outputs a `word`.

Reference: Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention): [link](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/).

## The Illustrated Transformer
Start with [Wikipedia](https://en.wikipedia.org/wiki/Attention_(machine_learning)).    
Cohere has a good example-based [page](https://docs.cohere.com/docs/the-attention-mechanism) on the Attention mechanism.

[link](https://jalammar.github.io/illustrated-transformer/)

## The Illustrated BERT, ELMo, and co.
[link](https://jalammar.github.io/illustrated-bert/)

## The Illustrated GPT-2 (Visualizing Transformer Language Models)
[link](https://jalammar.github.io/illustrated-gpt2)

## A Visual Guide to Using BERT for the First Time
[link](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

## How GPT3 Works - Visualizations and Animations
[link](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

## Interfaces for Explaining Transformer Language Models
[link](https://jalammar.github.io/explaining-transformers/)

## The Illustrated Retrieval Transformer
[link](https://jalammar.github.io/illustrated-retrieval-transformer/)

## 
[link]()

## 
[link]()

## 
[link]()





