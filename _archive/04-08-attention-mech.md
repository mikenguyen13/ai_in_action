---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

## The Rise of Attention 


> After transforming raw text into a clean, tokenized corpus and mapped each token to a learnable embedding vector.  The result is a matrix
>
> $$X = [\mathbf{x}_1,\dots,\mathbf{x}_L] \in \mathbb{R}^{L\times d},$$
>
> which contains *what* the model should know about the sequence, but not *how* the tokens interrelate.  Our next task is to endow these vectors with *context*.

A classic solution is to process $X$ sequentially with a recurrent function

$$
\mathbf{h}_t = f\bigl(\mathbf{x}_t,\mathbf{h}_{t-1},\theta\bigr), \qquad t = 1,\dots,L,
$$

so that $\mathbf{h}_t$ summarizes *all* tokens up to position $t$.  Gated cells such as **LSTM** \[Hochreiter & Schmidhuber 1997] and **GRU** {cite}`cho2014learning` improve gradient flow, yet every step still compresses the entire past into a single state vector.

Thus, one vector must serve two opposing roles:

1. **Storage**: remember information that *might* matter later.
2. **Retrieval**: expose exactly the pieces required *now*.

The conflict surfaces mathematically in the gradient path length.  If $\ell(\theta)$ is the loss at time $L$, the influence of an early token $\mathbf{x}_k$ obeys

$$
\frac{\partial \ell}{\partial \mathbf{x}_k}
   = \frac{\partial \ell}{\partial \mathbf{h}_L}
     \prod_{t=k+1}^{L} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}},
$$

whose norm decays or explodes exponentially with $(L-k)$ when eigenvalues of the transition Jacobian depart from 1.  Empirically, gated RNNs cope up to roughly 50 tokens {cite}`pascanu2013difficulty`.  Beyond that, long‑range dependencies degrade.

> **Take‑away** Sequential compression is a bottleneck.  We need a mechanism that can *store* all tokens yet *retrieve* arbitrary ones on demand.

---

### 1 Alignment — The Historical Spark

**Bahdanau, Cho, & Bengio (2015)** introduced *content‑based addressing* for neural machine translation:

$$
\boxed{\alpha_{tj}=\operatorname{softmax}_{j}(e_{tj})},\qquad e_{tj}=g\bigl(\mathbf{s}_{t-1}^{\text{dec}},\,\mathbf{h}_{j}^{\text{enc}}\bigr).
$$

The **context vector**

$$
\mathbf{c}_t = \sum_{j=1}^{L} \alpha_{tj}\,\mathbf{h}_{j}^{\text{enc}}
$$

offers *random‑access* retrieval over the entire source sentence, sidestepping the single‑vector bottleneck.  Dot‑product scoring \[Luong et al. 2015] and later generalization beyond encoder–decoder settings \[Vaswani et al. 2017] paved the way for the Transformer.

---

### 2 General Formulation of Attention

Given batched representations

$$
\mathbf{Q}\in\mathbb{R}^{B\times L_q\times d_k},\mathbf{K}\in\mathbb{R}^{B\times L_k\times d_k},\mathbf{V}\in\mathbb{R}^{B\times L_k\times d_v},
$$

the **scaled dot‑product attention** is

$$
\operatorname{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})
 = \operatorname{softmax}\!\Bigl(\tfrac{\mathbf{Q}\mathbf{K}^{\!\top}}{\sqrt{d_k}}\Bigr)\mathbf{V}.
$$

* Each query vector scores similarity against *all* keys.
* Division by \$\sqrt{d\_k}\$ stabilizes gradients.
* Softmax along the key dimension turns scores into a probability simplex.

---

### 3 Four Canonical Variants

| Variant                       | Score Matrix \$S\$                                      | Mask \$M\$                  | Typical use‑case        |
| ----------------------------- | ------------------------------------------------------- | --------------------------- | ----------------------- |
| **(a) Simplified self‑attn**  | \$S=\mathbf{Q}\mathbf{K}^{!\top}\$                      | none                        | pedagogical baseline    |
| **(b) Scaled self‑attn**      | \$S=\dfrac{\mathbf{Q}\mathbf{K}^{!\top}}{\sqrt{d\_k}}\$ | optional padding mask       | encoder layers          |
| **(c) Causal attn**           | same as (b)                                             | \$M\_{ij}=1\$ if \$j>i\$    | autoregressive decoding |
| **(d) Multi‑head attn (MHA)** | concat of \$H\$ sub‑scores \$S^{(h)}\$                  | per‑head mask as in (b)/(c) | modern LLMs             |

Masked softmax sets \$\alpha\_{ij}=0\$ where \$M\_{ij}=1\$ by subtracting a large constant before softmax.

---

### 4 Illustrative Numerical Example

Consider the sentence “**The · book · was · fascinating**” embedded in two dimensions:

$$
\mathbf{X}=\begin{bmatrix}0.6&0.2\\0.9&1.1\\0.1&1.2\\0.3&0.5\end{bmatrix}.
$$

With \$\mathbf{Q}=\mathbf{K}=\mathbf{V}=\mathbf{X}\$ the raw similarity matrix is

$$
S=\begin{bmatrix}
0.40&1.26&0.84&0.33\\
1.26&2.02&1.43&0.96\\
0.84&1.43&1.45&0.81\\
0.33&0.96&0.81&0.34\end{bmatrix}.
$$

Applying **variant (a)** yields

$$
\alpha_{1{:}}=[0.141,0.423,0.288,0.148],\qquad
\mathbf{y}_1=\begin{bmatrix}0.71\\0.75\end{bmatrix}.
$$

Scaled, causal, and multi‑head variants modify either the logits, the accessible positions, or the representation subspace, but the core operation is unchanged.

---

### 5 Computational Complexity

| Model                   | Time / layer  | Memory       | Test‑time KV cache |
| ----------------------- | ------------- | ------------ | ------------------ |
| RNN / LSTM              | \$O(Ld^2)\$   | \$O(d)\$     | N/A                |
| Self‑attention          | \$O(L^{2}d)\$ | \$O(L^{2})\$ | ✓                  |
| Causal self‑attn (inf.) | \$O(Ld)\$     | \$O(Ld)\$    | ✓                  |

FlashAttention‑2 reduces memory to \$O(Ld)\$ during training by tiling into GPU SRAM \[Dao et al. 2023].

---

### 6 Qualitative Insights

* **Content‑based invariance** — similarity, not absolute position, drives retrieval.
* **Head specialization** — individual heads discover grammar, coreference, or positional rules \[Clark et al. 2019].
* **Causality via masking** — encoder and decoder blocks share weights; only the mask differs.

---

### 7 Key Takeaways and Forward Pointer

1. We now possess a vocabulary of attention variants that disentangle *storage* and *retrieval*.
2. Scaling, masking, and head‑splitting are independent design knobs underpinning modern Transformers.
3. The same core equation scales from toy sentences to billion‑parameter LLMs.

In **Chapter Y** we will translate these equations into optimized PyTorch code and benchmark each variant inside a causal LLM block.

---

### References

* Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
* Cho, K., van Merriënboer, B., Gulcehre, C., et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.* EMNLP.
* Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). *What Does BERT Look at?* ACL.
* Dao, T., et al. (2023). *FlashAttention‑2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691.
* Hochreiter, S., & Schmidhuber, J. (1997). *Long Short‑Term Memory.* *Neural Computation* 9(8), 1735‑1780.
* Luong, M.‑T., Pham, H., & Manning, C. D. (2015). *Effective Approaches to Attention‑based Neural Machine Translation.* EMNLP.
* Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the Difficulty of Training Recurrent Neural Networks.* ICML.
* Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.

---


## Capturing Data Dependencies with the Attention Mechanism

Recurrent Neural Networks (RNNs) have been successfully applied to sequence modeling tasks such as machine translation. However, they face limitations when dealing with long sequences. One significant shortcoming is that the entire input sequence must be compressed into a fixed-size hidden state before being passed to the decoder. This bottleneck restricts the ability of the model to retain and access long-range dependencies effectively.

To address this limitation, Bahdanau et al. (2014) proposed an attention mechanism that allows the decoder to selectively access different parts of the input sequence at each decoding step. This enhancement significantly improved the performance of RNN-based encoder-decoder architectures in tasks like translation.

Remarkably, just a few years later, Vaswani et al. (2017) demonstrated that RNNs are not required for natural language processing. They introduced the Transformer architecture, which relies entirely on a self-attention mechanism—a concept inspired by Bahdanau’s earlier work. Self-attention now forms the cornerstone of all modern large language models (LLMs), including the GPT series.

## Attending to Different Parts of Inputs with Self-Attention

Let us explore the inner workings of self-attention and implement a simplified version from scratch. This foundational understanding is critical for mastering the Transformer architecture and, by extension, the structure of modern LLMs.

Self-attention computes attention weights by relating different positions within a single input sequence. Unlike traditional attention mechanisms, which align elements between two sequences (e.g., source and target in translation), self-attention focuses exclusively on intra-sequence relationships—between tokens within the same sentence, for example.

## A Simplified Self-Attention Implementation (No Trainable Weights)

To concretely understand self-attention, we will build a simplified model in PyTorch that operates without trainable weights.

We begin with a sequence of input embeddings, where each token in a sentence is mapped to a fixed-dimensional vector. Assume the inputs are three 3-dimensional vectors:

```python
import torch

# Example input: three 3-dimensional token embeddings
inputs = torch.tensor([
    [1.0, 0.0, 1.0],  # x1
    [0.0, 1.0, 1.0],  # x2
    [1.0, 1.0, 0.0]   # x3
])
```

### Step 1: Compute Raw Attention Scores

We compute the raw attention scores using dot products between a chosen query vector and every input vector. For example, using `x1` as the query:

```python
query = inputs[0]
attention_scores = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attention_scores[i] = torch.dot(query, x_i)

print("Attention Scores:", attention_scores)
```

**Dot Product Interpretation**:
The dot product between vectors quantifies their alignment. A higher score implies greater similarity or relevance between the query and a specific input vector.

### Step 2: Normalize the Scores to Obtain Attention Weights

To make attention weights interpretable and ensure they sum to 1, we normalize the scores:

```python
attention_weights = attention_scores / attention_scores.sum()
print("Normalized Weights:", attention_weights)
print("Sum:", attention_weights.sum())
```

However, in practice, the **softmax function** is preferred:

```python
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attention_weights = softmax(attention_scores)
print("Softmax Weights:", attention_weights)
print("Sum:", attention_weights.sum())
```

Using softmax not only normalizes the weights but ensures they are positive and numerically stable.

Alternatively, use PyTorch's optimized implementation:

```python
import torch.nn.functional as F

attention_weights = F.softmax(attention_scores, dim=0)
print("PyTorch Softmax Weights:", attention_weights)
print("Sum:", attention_weights.sum())
```

### Step 3: Compute the Context Vector

The final context vector `z₁` for query `x₁` is computed as a weighted sum of all input vectors:

```python
context_vector = torch.zeros_like(query)

for i, x_i in enumerate(inputs):
    context_vector += attention_weights[i] * x_i

print("Context Vector z₁:", context_vector)
```

The context vector is an enriched embedding of the query token, incorporating relevant information from the entire sequence. This mechanism is essential for contextualizing token representations in modern LLMs.


