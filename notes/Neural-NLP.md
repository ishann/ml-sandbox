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

Transformers are encoder-decoder NMTs which benefit from parallelization of the attention mechanism.

Transformers are especialy good at modeling context using attention (from surrounding words/ sentences/ paragraphs) to decipher the semantics associated with a word. For example, for context resolution between "run!" v/s. "run a machine" v/s. "a long run of success".    
Thus, attention help us move beyond _lexicographic_ similarity towards _semantic_ similarity: $\text{embeddings}\rightarrow\text{contextual embeddings}$

Transformers are a type of deep learning model that have revolutionized various natural language processing tasks. They work by processing input sequences and generating output sequences through a mechanism called self-attention, which allows them to capture relationships between words or tokens in the input. 

High-level summary of how transformers work:
1. Input Representation:
   - Let's assume we have an input sequence of tokens, where each token is represented as a vector. These input vectors are denoted as $X=[x_1, x_1, ..., x_n]$, where $n$ is the sequence length.
2. Self-Attention Mechanism:
   - For each token $x_i$, compute attention scores wrt all other tokens in the sequence. These _attention_ scores are computed as follows:    
   $\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$    
   where,    
     - $Q$, $K$, and $V$ are matrices obtained by linear transformations of the input X.
       - $Q$ represents the query vectors,
       - $K$ represents the key vectors, and 
       - $V$ represents the value vectors.
     - $d_k$ is the dimensionality of the key vectors.
   - The softmax normalized attention scores are used to weigh the value vectors, allowing the model to assign varying importance to different tokens in the input sequence when generating the output.
3. Multi-Head Attention:
   - To capture different types of relationships between tokens, transformers use multiple attention heads to expand the model’s ability to focus on different positions.
   - Thus, multi-head attention results in multiple parallel _representation subspaces_.
   - Each head has its own sets of $Q$, $K$, and $V$ matrices, and the results from all heads are concatenated and linearly transformed to obtain the final output.
4. Positional Encoding:
   - Transformers do not have a built-in notion of sequence order, so they use positional encodings to provide information about the positions of tokens in the input sequence. These positional encodings are added to the input embeddings: $X'=X+PE$
5. Encoder and Decoder Stacks:
   - Each encoder block (self-attention $\rightarrow$ ffnn) as well as decoder block (self-attention $\rightarrow$ encoder-decoder attention $\rightarrow$ ffnn) sub-layer  has a residual connection and is followed by a layer-normalization step.
   - Transformers consist of multiple layers of encoders and decoders, which stack
     - self-attention (both, but decoder only attends to earlier positions masking future with $-\infty$)
     - encoder-decoder attention (decoder only -  get $Q$ from the layer below, and take $K$, $V$ from encoder stack output)
     - feedforward (both) layers.
6. Feedforward Layer:
   - After the self-attention mechanism, a feedforward neural network is applied to each position independently:    
     $\text{FFNN}(X) = W_2^T\text{ReLU}(W_1^TX + b_1)+b_2$, where $X$ is the output from the self-attention layer and $W_i$/ $b_i$ are learnable parameters.
7. Output Generation:
   - The final output of the transformer is obtained from the decoder's stack.
   - A linear transformation followed by softmax produces output probabilities over the vocabulary:    
   $\text{Output} = \text{softmax}(W^TX+b)$, where $X$ is the output from the decoder stack and $W$ and $b$ are learnable parameters.


References:
1. The Illustrated Transformer: [link](https://jalammar.github.io/illustrated-transformer/).
2. ChatGPT.

## The Illustrated BERT, ELMo, and co.


References:
1. [link](https://jalammar.github.io/illustrated-bert/)
2. ChatGPT.

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


