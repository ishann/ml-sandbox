## Attention Is All You Need

Transformers can be thought of as a new `op` which  replaces recurrence and convolution with attention mechanisms. A scaled dot-product attention unit, extended via multi-head attention captures diverse relationships in parallel. Positional encodings encode sequence order information. Stacked encoder and decoder layers interleave `(1)` attention, `(2)` feed-forward sublayers, `(3)` residual connections, and `(4)` layer normalization to build compositional representations. Masking - in the decoder - prevents future information leakage (though bidirectional models do not preclude bidirectional dependencies at train time).


### Scaled Dot-Product Attention

Scaled dot-product attention computes how each element in a sequence should attend to every other element. It transforms the input embeddings into three distinct spaces — queries ($Q$), keys ($K$), and values ($V$). These spaces are then scored and weighted to produce context-aware representations.

Given query **Q**, key **K**, and value **V** matrices, attention is computed as
$$\text{Attention}(Q,K,V) = \mathrm{softmax}\!\bigl(\tfrac{QK^\top}{\sqrt{d_k}}\bigr)\,V.$$

* **Query–Key Matching**: The query asks the question: *What information do I need?*. Keys are potential answers. Attention scores tell us which keys best match our query. The matrix multiplication $QK^\top$ produces raw similarity scores between queries and keys.
* **Weighted Context**: The resulting softmax weights act like a “soft alignment” over positions, producing a context vector that blends information from all tokens according to relevance.
* Dividing by $\sqrt{d_k}$ stabilizes gradients. Without scaling, dot products grow large in high dimensions, pushing softmax into regions with extremely small gradients. The $\sqrt{d_k}$ normalization keeps the logits in a more moderate range, allowing each token to gather information from all others.


### Multi-Head Attention

Rather than a single attention head, the Transformer uses $h$ parallel heads, each with its own learned linear projections for $Q$, $K$, and $V$. Heads run simultaneously to capture different types of relationships (e.g., syntax vs. semantics), then concatenate their outputs and linearly project back to the model space. This enriches representation power without significant computational overheads.

1. Project $Q,K,V$ into $h$ subspaces via linear layers.
2. Compute scaled dot-product attention independently for each head.
3. Concatenate the $h$ head outputs and linearly project back to the model dimension.

This allows the model to attend jointly to information from different representation subspaces at different positions.


### Positional Encoding

Transformers apply self-attention in parallel across tokens, making them inherently permutation-invariant. Without encoding any position information, the sequence $\texttt{a,b,c,d}$ and the sequence $\texttt{d,c,a,b}$ would produce identical attention patterns. But $\texttt{seq2seq}$ models need to be aware of order. Positional encodings remedy this by giving each token a unique position signal added to its embedding before attention layers.

The original paper adds sinusoidal functions of differing frequencies that are summed with token embeddings. Alternative approaches include learned positional embeddings (where positions are end2end trained vectors) used in BERT and GPT models, and Rotary Positional Embeddings (where position is encoded via rotations in embedding space) used in LLaMA and PALM.


### Encoder & Decoder Stacks

* **Encoder**: Composed of $N$ identical layers, each with two sub-layers:

  1. Multi-head self-attention over the input tokens.
  2. Position-wise fully connected feed-forward network.

* **Decoder**: Also $N$ layers, but each layer has three sub-layers:

  1. Masked multi-head self-attention (masking preserves the autoregressive property of not looking at future tokens).
  2. Multi-head attention over encoder outputs (allowing the decoder to “look” at the source).
  3. Feed-forward network.

In both encoders and decoders, residual connections around each sub-layer and layer normalization ensure stable training.


### Feed-Forward Networks

Each position-wise feed-forward network takes the form:
$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)\,W_2 + b_2.
$$

Since FFNs consist of fully connected layers, they make up the majority of model parameters.

#### Miscellaneous

* **Time Complexity**: Time complexity per layer per token is $O(n^2d)$ due to $QK^\top$ dot products (with sequence length $n$ and model dimension $d$).
* **Memory complexity**: Model complexity is $O(n^2)$, limiting very long sequences without modifications (cf. sparse attention).
* In the Transformer’s **encoder** and **decoder** layers:
  * **Self-Attention**: Queries, keys, and values all come from the same sequence (enable each position to attend to every other).
  * **Encoder–Decoder Attention**: In the decoder, queries come from the previous decoder layer, while keys and values come from the encoder output (allowing the decoder to look back at the source).
  * **Masking**: In the decoder’s self-attention, a causal mask prevents attending to future tokens, preserving autoregressive decoding.


#### References

1. Attention Is All You Need. [arxiv](https://arxiv.org/abs/1706.03762), [google-blog](https://research.google/pubs/attention-is-all-you-need).
2. Transformer: A Novel Neural Network Architecture for Language Understanding. [google-blog](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding).


