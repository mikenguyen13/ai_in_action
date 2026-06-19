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


# Walk-Through of Scaled Dot-Product Self-Attention

## 1 Motivation and Intuition

The transformer architecture introduced the **self-attention** mechanism to let every token in a sequence “look at” every other token and decide, in a data-dependent way, how much each of them should influence its own representation \:cite:`Vaswani2017`. Unlike recurrent networks, self-attention is **order-agnostic** and highly parallelizable.

To keep the mental model concrete, imagine that you are skimming a paragraph and you pause on the word *it*. You instinctively glance backward to decide what *it* refers to; the words you revisit receive higher “attention.” Queries, keys and values are simply the mathematics that formalize this glance.

```{admonition} Why the database metaphor?
:class: tip
In information-retrieval terminology  
* **Query** – what you are looking for.  
* **Key** – an index used to match the query.  
* **Value** – the payload you actually want once a key matches.  
Self-attention re-uses the same idea: every token *broadcasts* a key and a value, and *asks* with its query which keys (hence which tokens) matter.
```

---

## 2 Notation at a Glance

| Symbol                                     | Shape                                | Meaning                       |
| ------------------------------------------ | ------------------------------------ | ----------------------------- |
| $T$                                      | —                                    | Sequence length               |
| $d_{\text{in}}$                         | —                                    | Input embedding dimension     |
| $d_{\text{out}}$                        | —                                    | Projection (output) dimension |
| $\mathbf X$                              | $(T,d_{\text{in}})$               | Input embeddings              |
| $\mathbf W_q,\mathbf W_k,\mathbf W_v$ | $(d_{\text{in}},d_{\text{out}})$ | Trainable projections         |
| $\mathbf Q = \mathbf X\mathbf W_q$      | $(T,d_{\text{out}})$              | Queries                       |
| $\mathbf K = \mathbf X\mathbf W_k$      | $(T,d_{\text{out}})$              | Keys                          |
| $\mathbf V = \mathbf X\mathbf W_v$      | $(T,d_{\text{out}})$              | Values                        |
| $\mathbf Z$                              | $(T,d_{\text{out}})$              | Context (output) vectors      |

---

## 3 Mathematical Formulation

For a token position $t\in{1,\dots,T}$ let

$$
\alpha_{ti}= 
\operatorname{softmax}_i
\Bigl(
\underbrace{\frac{\mathbf q_t\mathbf k_i^\top}
                 {\sqrt{d_{\text{out}}}}}_{\text{scaled dot-product}}
\Bigr),
\qquad
z_t \;=\;\sum_{i=1}^{T}\alpha_{ti}\,\mathbf v_i .
$$

* **Scaled dot-product.** Dividing by $\sqrt{d_{\text{out}}}$ prevents the dot products from growing with dimension, which would otherwise squash softmax gradients and slow training.
* **Softmax.** Normalizes scores into a probability distribution that sums to 1.
* **Weighted sum.** The context vector $z_t$ blends value vectors according to the attention weights $\alpha_{ti}$.

---

## 4 Step-by-Step Implementation in PyTorch

### 4.1 Toy setup

```{code-cell} python
import torch
torch.manual_seed(42)

T, d_in, d_out = 4, 4, 3                 # sequence length and dimensions
X = torch.randint(0, 10, (T, d_in)).float()  # toy input embeddings

W_q = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_k = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_v = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

X.shape, W_q.shape
```

> **Shapes check**
> *Input → (4 × 4), Weights → (4 × 3).*

### 4.2 Project to queries, keys and values

```{code-cell} python
Q = X @ W_q
K = X @ W_k
V = X @ W_v
Q.shape, K.shape, V.shape
```

### 4.3 Focus on a single position (index 1)

```{code-cell} python
t = 1                           # our “current” token
q_t = Q[t]                      # shape (d_out,)
scores_t = (q_t @ K.T) / d_out**0.5
weights_t = torch.softmax(scores_t, dim=-1)
z_t = weights_t @ V
z_t
```

### 4.4 Vectorized computation for all positions

```{code-cell} python
scores = (Q @ K.T) / d_out**0.5          # (T × T)
weights = torch.softmax(scores, dim=-1)  # (T × T)
Z = weights @ V                          # (T × d_out)
Z
```

```{admonition} Parameter weights ≠ attention weights
:class: note
*Parameter* matrices $\mathbf W_q,\mathbf W_k,\mathbf W_v$ are static during one forward pass and learned over many updates.  
*Attention* weights $\alpha_{ti}$ are **dynamic** – they are re-computed every time new input arrives.
```

---

## 5 Worked Numerical Example

Below is the complete computation for the toy sequence. Feel free to run the code in Jupyter Book and inspect intermediate tensors.

```{code-cell} python
with torch.no_grad():
    for name, tensor in zip(
        ["X", "Q", "K", "V", "Attention Weights", "Context Z"],
        [X, Q, K, V, weights, Z]
    ):
        print(f"\n{name}:\n{tensor.round(3)}")
```

---

## 6 Visualizing an Attention Map

```{code-cell} python
import matplotlib.pyplot as plt

plt.imshow(weights, interpolation="nearest")
plt.xlabel("Key index $i$")
plt.ylabel("Query index $t$")
plt.title("Attention weights $\\alpha_{ti}$")
plt.colorbar(label="Weight")
plt.show()
```

The heatmap highlights which tokens dominate the context of each query position.

---

## 7 From Single Head to Multi-Head

In practice transformers use $h$ parallel heads:

1. Split $d_{\text{out}}$ into $h$ smaller sub-spaces.
2. Repeat the above computation in each sub-space.
3. Concatenate the resulting contexts along the feature dimension.

Multi-head attention lets the model attend to heterogeneous relationships simultaneously.

---

## 8 Key Takeaways

* Self-attention positions each token in *conversation* with every other token.
* Scaling by $\sqrt{d_{\text{out}}}$ keeps gradients healthy as dimensions grow.
* Queries, keys and values are *projections* – not mysterious new entities – learned end-to-end with the rest of the network.
* The same mechanics extend seamlessly to multi-head and masked attention used in large language models \:cite:`Devlin2019,Brown2020`.

