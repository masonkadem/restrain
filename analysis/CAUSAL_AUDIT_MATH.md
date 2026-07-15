# Causal-use audit: precise formulation

This note states exactly what the toy in `analysis/causal_mediation_toy.py`
defines and measures. It is deliberately black-and-white: every object below is
computed in the code, and every audit is a function of a *frozen* model plus the
governing equation — no held-out measured labels are used to reach a verdict.

Notation: $p$ positional dimension, $w$ attention width, $S$ number of slots.
Vectors are rows; $[\,\cdot\,;\,\cdot\,]$ is concatenation.

---

## 1. Task and governing law

A **governing law** is a known function $g:\mathbb{R}^2\to\mathbb{R}$ of two
components, together with input ranges. Each example draws a bank of $S$ slot
values $v_1,\dots,v_S$ and a query naming two slots $i,j$. The target is

$$ y \;=\; g(v_i, v_j). $$

Two instances are implemented:

- **Blood pressure (Moens–Korteweg).** $v_i=\text{PTT}$ (pulse transit time),
  $v_j=E_0$ (per-subject arterial-stiffness calibration),

$$ y \;=\; g(\text{PTT},E_0)\;=\;2\ln\!\frac{L}{\text{PTT}}\;-\;\ln\!\frac{E_0}{E_{\text{ref}}},
\qquad BP \;=\; P_0 + S\cdot y . $$

  Uncalibrated PTT does not determine $BP$: $y$ genuinely depends on $E_0$.

- **Saturation (Beer–Lambert).** $y=g(v_i,v_j)=v_i/(v_i+v_j)$.

The two components are the "two things a transformer should encode." A model is
*trustworthy on an input* iff its output is produced **through** $g$ — using both
components in the way $g$ prescribes — not through a correlate that happens to
track $y$ on the training distribution.

---

## 2. Architecture (two-query cross-attention retriever)

Input encodings: the source $s\in\mathbb{R}^{2p}$ is the concatenation of the
positional codes of slots $i$ and $j$; the target matrix
$T\in\mathbb{R}^{S\times(1+p)}$ stacks, per slot, a (standardized) value channel
and its positional code. All maps $W_\bullet$ are 2-layer MLPs.

$$
q_i = W_Q^{i}(s_{1:p}),\qquad q_j = W_Q^{j}(s_{p+1:2p}),\qquad
K = W_K(T),\quad V = W_V(T)\in\mathbb{R}^{S\times w}.
$$

$$
a_i=\operatorname{softmax}\!\Big(\tfrac{q_iK^\top}{\sqrt w}\Big),\quad
c_i=a_iV\in\mathbb{R}^{w};\qquad
a_j=\operatorname{softmax}\!\Big(\tfrac{q_jK^\top}{\sqrt w}\Big),\quad
c_j=a_jV\in\mathbb{R}^{w}.
$$

$$
\boxed{\,h=[\,c_i\,;\,c_j\,]\in\mathbb{R}^{2w}\,}\qquad
\hat y = W_O(h)\in\mathbb{R}.
$$

$h$ is the **audited state**. Two separate queries are required because a single
softmax attention returns a convex combination of values (adequate for the
averaging task) but cannot place the two components in separate coordinates,
which the ratio/log laws need. Training minimizes
$\frac1N\sum_n (\hat y_n - t_n)^2$ where $t_n$ is the training target (below);
weights are then **frozen** for all audits.

---

## 3. The three models (same architecture, same task)

They differ only in the training distribution of $v_j$ and the training target
$t$. Let $g_{\text{sur}}$ be a **wrong-functional-form** surrogate that agrees
with $g$ at $v_j=v_j^{\text{ref}}$ and to first order, but diverges away from it
(for BP, $g_{\text{sur}} = 2\ln(L/\text{PTT}) + \ln(E_0/E_{\text{ref}})$ — the
calibration term with the wrong sign, hence the same $|\partial/\partial v_j|$).

| model | train $v_j$ | train target $t$ | intended behavior |
|-------|-------------|------------------|-------------------|
| **law** | varies | $g(v_i,v_j)$ | uses both components, true law |
| **unfaithful** | varies | $g_{\text{sur}}(v_i,v_j)$ | uses $v_j$, **wrong** law |
| **shortcut** | $\approx$ const | $g(v_i,v_j)$ | ignores $v_j$ |

Evaluation distributions (all vs. the *true* $g$):

- **validation** $\mathcal D_{\text{val}}$: $v_j\approx v_j^{\text{ref}}$ — the
  narrow distribution a practitioner has. Here $g_{\text{sur}}\approx g$, so all
  three models look correct.
- **audit** $\mathcal D_{\text{aud}}$ and **OOD** $\mathcal D_{\text{ood}}$: $v_j$
  varies over its full range.

---

## 4. Conventional checks (what a practitioner would compute)

**Predictive error.** $\ \mathrm{MSE} = \mathbb E_{\mathcal D}\,(\hat y - y)^2.$
On $\mathcal D_{\text{val}}$ all three models have small MSE.

**Decodability probe.** Fit ridge weights on frozen states,

$$ \hat\beta=\arg\min_\beta \sum_n \| \beta^\top h_n - y_n\|^2 + \lambda\|\beta\|^2,
\qquad R^2\big(y,\hat\beta^\top h\big)\ \text{on held-out}\ \mathcal D_{\text{val}}. $$

High $R^2$ for all three: the sufficient statistic is *present* in $h$. This is
the key negative — **decodability does not imply causal use.**

---

## 5. Audit 1 — counterfactual sensitivity (input intervention)

For each audit input, compare the model's local sensitivity to component $j$
against the equation's, using autograd:

$$ \sigma_{\text{model}} = \mathbb E_{\mathcal D_{\text{aud}}}\Big|\tfrac{\partial \hat y}{\partial v_j}\Big|,
\qquad
\sigma_{g} = \mathbb E_{\mathcal D_{\text{aud}}}\Big|\tfrac{\partial g}{\partial v_j}\Big|
\quad(\text{for BP: } |\partial g/\partial E_0| = 1/E_0). $$

Reported score $\min(\sigma_{\text{model}}/\sigma_g,\,1)\in[0,1]$. The **shortcut**
model scores $\approx 0$; the **law** and **unfaithful** models both score
$\approx 1$. A gradient-*magnitude* check therefore **cannot** separate the
faithful model from the wrong-sign one — this is exactly why a representation
audit is needed.

---

## 6. Audit 2 — interchange intervention / DAS (representation intervention)

This asks whether a subspace of $h$ **causally carries** $y$ in the sense of
causal abstraction. Learn an orthonormal $R\in\mathbb{R}^{2w\times 2w}$ (frozen
model). For a base input $b$ and source input $s$, rotate, swap the top-$k$
coordinates, rotate back, and read the frozen head:

$$
z^{(b)}=R^\top h^{(b)},\quad z^{(s)}=R^\top h^{(s)},\qquad
\tilde z=\big[\,z^{(s)}_{1:k}\,;\,z^{(b)}_{k+1:2w}\,\big],\qquad
\hat y_{\text{patch}} = W_O\!\big(R\,\tilde z\big).
$$

$R$ is trained to make the intervened output equal the **equation's**
counterfactual — with $y=g$ the sufficient statistic, that target is the
source's value $y^{(s)}$:

$$
R^\star=\arg\min_R\ \mathbb E_{b,s}\ \| \hat y_{\text{patch}}(b,s) - y^{(s)}\|^2,
\qquad
\mathrm{IIA}(k)=1-\frac{\mathbb E_{b,s}\,\|\hat y_{\text{patch}}-y^{(s)}\|^2}
{\operatorname{Var}(y^{(s)})}.
$$

Interpretation: $\mathrm{IIA}=1$ means a $k$-dim subspace of $h$ is a faithful
causal variable for $y$ (interchanging it transplants the answer as $g$
dictates). Results (BP, mean over seeds): **law** $\approx 0.99$, **shortcut**
$\approx 0.2$, **unfaithful** $<0$ (below chance — its calibration is
anti-correlated with $g$). Only this audit isolates the faithful model.

Faithfulness floor: on **clean** inputs, patching a known $y$ must reproduce
$g$'s answer before the audit is trusted on corrupted inputs.

---

## 7. Confirmation (revealed after the audits)

$\mathrm{MSE}$ on $\mathcal D_{\text{ood}}$ (full $v_j$ range) confirms the
verdict the audits reached *without these labels*: **law** $\approx 0$,
**shortcut** intermediate, **unfaithful** the worst. The audit thus stands in for
OOD labels the practitioner does not have; the governing equation is the oracle.

---

## 8. One-line thesis and the open theory question

> A governing equation specifies a causal audit: on a frozen model, an
> interchange intervention that transplants the equation's sufficient statistic
> certifies causal use where predictive error, decodability, and gradient
> sensitivity all fail.

Open question worth a proposition: give conditions under which a representation
is linearly decodable for a quantity it does **not** causally use (Sections 4 vs
6 exhibit it empirically), and bound OOD robustness on the law's manifold below
by $\mathrm{IIA}$.
