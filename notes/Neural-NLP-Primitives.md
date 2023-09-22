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

## The Illustrated BERT, ELMo, and co. (Transfer Learning in NLP)

What makes BERT NLP's AlexNet+ImageNet moment: An outrageously large model is pre-trained on an outrageous amount of data, enabling generalized language understanding.
1. Semi-supervised learning: BERT is pre-trained on a large amount of unlabelled data.
2. Transfer learning: Fine-tuned on specific tasks to achieve SoTA on several benchmarks.
3. The availability of pre-trained model parameters and training code makes it accessible to lay-people. 

### Word Embeddings
* Training the specialized $[\text{CLS}]$ token's embedding enabled BERT-scale generalized language sentence embeddings, which are quite useful for training simple and effective linear classifiers for off-the-shelf classification tasks.
* ELMo: Context Matters
  * GloVe and Word2Vec embeddings are context-free, i.e., they generate a distinct embedding for each word in the vocabulary.
  * ELMo gained its language understanding super-powers by learning to predict the next word in a sequence (which can be trained using _massive_ amounts of online text data in an unsupervised manner).
  * ELMo looks at the entire sentence, using a bi-directional LSTM (i.e., looking both ways), to compute word embeddings:
    * Let input sequence be $\{x_1, x_2, ..., x_T\}$, where $T$ is the length of the input sequence.
    * Bidirectional LSTM capture contextual information in both directions for each word.    
      Let $h_f(t) = LSTM(x_t, h_f(t-1))$ and $h_b(t) = LSTM(x_t, h_b(t+1))$ be the forward and backward LSTMs, where:
       - $x_t$ is the input word embedding at position $t$.
       - $h_f(t)$ represents the hidden state of the forward LSTM at position $t$.
       - $h_b(t)$ represents the hidden state of the backward LSTM at position $t$.
    * Combining Contextual Representations: Concatenate the forward and backward hidden state activations at each biLSTM layer, and compute a linear combination with trainable weights:    
      $E(t) = \gamma \cdot \sum_i^L s_i \cdot [h_f^i(t) || h_b^i(t)]$, where:
       - $E(t)$ is the contextualized word embedding for $x_t$.
       - $\gamma$ is a scalar (task-dependent) weight parameter.
       - $s_i$ is a softmax normalized weight parameter to obtain a weighted sum over layer activations.
       - $L$ is the number of layers in the BiLSTM.
       - $h_f^i(t)$ and $h_b^i(t)$ are the forward and backward hidden state activations at layer $i$ for $x_t$.
    * Word Embedding: The final embedding is a combination of the BiLSTM and the original word embedding:    
      $ELMo(t) = E(t) + x_t$

### Generative Pre-Trained Transformers

Stacks decoders on top of decoders and trains in autoregressive manner, generating text in forward direction one token at a time. 

* Lets go off the Transformer encoder-decoder stack, and works with only the transformer decoder. The decoder is a natural choice for language modeling (predicting the next word) since it’s built to mask future tokens – a valuable feature when it’s generating a translation word by word.
* GPT-1 stacked 12 decoder layers, which trains well using only the vanilla self-attention from the decoder layers (with masking to avoid learning from future tokens).
* Predicting the next word on WebText data using the simpler Decoder stack results in a model that generates coherent  text. This is why the GPT family is exceptionally good at generating long-form coherent text.
* In constrast, BERT employs bidirectional context, and is trained on masked language modeling. Bidirectional modeling captures deeper contextual information, enabling BERT to perform well on a variety of downstream tasks.

**A few (more) things about BERT**:
* Masked language modeling is a clever way to train a bidirectional model in an autoregressive manner to learn deep contextual language modeling. Randomly mask $15\%$ of the input tokens and train the model to predict the masked words.
* Another task that BERT is pre-trained on next sentence prediction. Given two sentences, predict if the second sentence is the subsequent sentence in the original corpus.
* BERT for feature extraction: Pre-trained BERT can be used to generate contextualized word embeddings. The [CLS] token is especially useful for generating sentence embeddings.

**A few thoughts from Seb Ruder**:
* Pretrained word embeddings (word2vec/ GLoVe) only incorporate previous knowledge in the first model layer while the rest of the downstream network is trained from scratch. ELMo/ BERT/ GPT flipped this: use pre-trained embeddings with linear/ shallow networks trained from scratch.
* Just like ImageNet, pre-training on a large corpus of text enables the model to learn general language understanding, which can be fine-tuned on specific tasks with very little data.
* Language modeling is a good pre-training task.
  * Predicting the most probable next word requires an ability to express grammatical syntax and also model semantics.
  * It is an unsupervised and can be trained on massive amounts of data.
  * It has been shown to capture many facets of language relevant for downstream tasks, such as long-term dependencies, hierarchical relations, and sentiment.
  * Doing well on language modeling requires what could be considered _world knowledge_ or _common sense_.
  *  Opens the door to developing models for previously under-served languages. For very low-resource languages where even unlabeled data is scarce, multi-lingual language models may be trained on multiple related languages at once.
  * LLMs trained on language modeling as a pre-training task have been empirically shown to be extremely sample-efficient, achieving good performance with only hundreds of examples and are even able to perform zero-shot learning.


References:
1. The Illustrated BERT, ELMo, and co. [jay-alammar-blog](https://jalammar.github.io/illustrated-bert/).
2. NLP's ImageNet moment has arrived [seb-ruder-blog](https://ruder.io/nlp-imagenet/).
3. ChatGPT.

## The Illustrated GPT-2 (Visualizing Transformer Language Models)

 * GPT-2 is built using transformer decoder blocks. BERT uses transformer encoder blocks.
 * Both use absurd amounts of data and compute.
 * GPT2, like traditional language models, outputs one token at a time. After each token is produced, that token is added to the input sequence. The new       sequence becomes the input to the model in its next step (auto-regression).

 I understand the building blocks. Reading is becoming a distraction.
 Move on to [Zero to Hero](https://karpathy.ai/zero-to-hero.html).
 Revisit and go over remaining articles later: [illustrated-gpt2](https://jalammar.github.io/illustrated-gpt2), [how-gpt3-works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/), [explaining-transformers](https://jalammar.github.io/explaining-transformers/), [illustrated-retrieval-transformer](https://jalammar.github.io/illustrated-retrieval-transformer/).

