# ARC White-Box Estimation Challenge 2026 — Reference Dossier

**WhestBench 2026 — problem, theory, resources**
*Compiled 23 July 2026*

> **Given the weights of a neural network, can you predict its expected per-neuron activations more accurately than running it many times?** A working reference to the $100,000 contest run by the Alignment Research Center and AIcrowd — the exact task, the scoring machinery, the mathematics it rests on, measured baselines, and every source worth reading.

| | |
|---|---|
| **Organisers** | Alignment Research Center × AIcrowd |
| **Phase 1 closes** | 31 July 2026 (Phase 2: 1 Aug – 19 Sep) |
| **Prize pool** | $100,000 — $50k / $20k / $10k + $20k |
| **Configuration** | width n = 256, depth L = 8, ReLU, He init, CPU-only |

---

## Provenance — read this first

Parts 1–5 and 10–12 are drawn from the official challenge page, ARC's blog posts and the companion paper; these are **reported facts**. Parts 6–8 are **my own analysis and experiments**, run fresh for this document at the stated configuration — they are useful orientation, not organiser guidance, and you should re-derive anything you intend to rely on. Every measured figure in Part 6 comes from code you can reproduce in a few minutes.

---

## Contents

1. [At a glance](#part-01--at-a-glance)
2. [The problem, stated precisely](#part-02--the-problem-stated-precisely)
3. [Scoring, compute model, rules](#part-03--scoring-compute-model-rules)
4. [Why ARC is running this](#part-04--why-arc-is-running-this)
5. [State of the art: cumulant propagation](#part-05--state-of-the-art-cumulant-propagation)
6. [Measured baselines](#part-06--measured-baselines)
7. [Where the difficulty actually is](#part-07--where-the-difficulty-actually-is)
8. [An attack ladder](#part-08--an-attack-ladder)
9. [The mathematical toolkit](#part-09--the-mathematical-toolkit)
10. [Annotated resource index](#part-10--annotated-resource-index)
11. [Formula cheat sheet](#part-11--formula-cheat-sheet)
12. [Timeline, glossary, contacts](#part-12--timeline-glossary-contacts)

---

## Part 01 — At a glance

*The whole contest compressed into one page.*

| Field | Detail |
|---|---|
| **Full name** | ARC White-Box Estimation Challenge 2026 ("WhestBench") |
| **Run by** | Alignment Research Center × AIcrowd |
| **Committee** | Paul Christiano, Jacob Hilton, Wilson Wu (ARC); Sharada Mohanty, Dipam Chakraborty (AIcrowd) |
| **Task** | From weights alone, predict per-neuron expected post-ReLU activations |
| **Networks** | Randomly-initialised ReLU MLPs, He-Gaussian init, variance 2/n |
| **Configuration** | width n = 256, L = 8 hidden layers, input X ~ N(0, Iₙ) |
| **Output** | An L × n matrix; scored on the final-layer row only |
| **Budget** | ≈ 3.4 × 10¹⁰ analytical FLOPs per MLP |
| **Metric** | s = MSE_final × max(0.5, C/B), averaged over the suite. Lower is better. |
| **Submission** | Executable Python (`estimator.py`) in a tarball — not prediction files |
| **Environment** | Isolated CPU-only, pinned deps, no network access |
| **Prizes** | $50k / $20k / $10k for score; $20k for best algorithmic contribution |
| **Phase 1** | 18 June – 31 July 2026 |
| **Phase 2** | 1 August – 19 September 2026 (this decides prizes) |
| **Results** | 1 October 2026; NeurIPS Competition Track workshop 11–12 December (TBD) |
| **LLM use** | Explicitly permitted and encouraged |
| **Contact** | arc-whestbench@aicrowd.com · Discord: discord.gg/4gyQvzWPJ |

> **The one-sentence version.** A neural network is not a black box — you hold its weights — so an algorithm that reads the weights *ought* to beat one that can only watch outputs; ARC has proved this for wide random MLPs but their method degrades with depth, and this contest pays $100k to whoever fixes that.

### Why you should care beyond the prize

ARC's stated long-run question is whether a highly capable system would undermine human control in unusual situations. Running it on many inputs is unreliable, because a sufficiently capable system need not fall for honey-pots. So ARC wants estimates read *off the weights*. Doing that for trained frontier models is far out of reach; doing it for randomly-initialised MLPs is the tractable base case. This contest is that base case, made competitive.

---

## Part 02 — The problem, stated precisely

*Architecture, target quantity, and how ground truth is manufactured.*

### 2.1 The network

The estimator is handed the weights θ = (W⁽¹⁾, …, W⁽ᴸ⁾) of a plain ReLU MLP with no biases and no skip connections. Activations are defined layer by layer:

```
h⁽⁰⁾ = X
h⁽ˡ⁾ = ReLU( W⁽ˡ⁾ h⁽ˡ⁻¹⁾ )        for ℓ = 1, …, L

X  ~  N(0, Iₙ)                    # standard normal input
W⁽ˡ⁾ᵢⱼ ~ N(0, 2/n)  i.i.d.        # He initialisation
n = 256,  L = 8
```

> **A detail people miss.** ARC's paper and blog define the model with a *final linear layer* W⁽ᴸ⁺¹⁾ and ask for the expected network output. The contest omits that final linear layer and asks instead for the expected *post-ReLU activations*. ARC notes this makes essentially no difference to the problem, but it does change what your code must return — read the starter-kit contract, not the paper, for the exact shape.

### 2.2 The target

For each neuron i in each layer ℓ you must estimate

```
Ŷ_{ℓ,i}  ≈  E_{X ~ N(0, Iₙ)} [ h_i⁽ˡ⁾(X) ]
```

The expectation is over the *input distribution only* — the weights are fixed and given. This is worth internalising: you are not estimating an average over random networks, you are integrating one specific network against a Gaussian. Every weight matrix you receive is a different, fully-specified integration problem.

### 2.3 Ground truth

The expectation has no closed form. The organisers approximate it by pushing a very large pool of standard-normal samples through the network and averaging post-ReLU activations layer by layer. That Monte-Carlo reference *is* the ground truth you are scored against. Two consequences worth holding onto:

- Your target is a high-precision MC estimate, not the platonic integral. At sufficient accuracy you would be fitting their reference noise — though the reference pool is large enough that this is far below the current frontier.
- Only the **final-layer row** (ℓ = L) enters the leaderboard score. The earlier rows are reported as a diagnostic showing where approximation error accumulates across layers — which is a strong hint about what the organisers expect to go wrong.

### 2.4 What makes it hard, structurally

If the pre-activation Zᵢ⁽ᴸ⁾ were exactly Gaussian with mean μ and variance σ², the answer would be a closed form (see Part 11). So the entire problem reduces to two things: **(a)** propagating means and covariances accurately through eight ReLU layers, and **(b)** correcting for the fact that the activation distributions are *not* Gaussian.

Step (a) is not local. The variance of neuron i at layer L is wᵢᵀ Σ⁽ᴸ⁻¹⁾ wᵢ, which needs the *full* n × n covariance of the previous layer, including off-diagonals. Those off-diagonals need pairwise ReLU second moments, which need the joint distribution of pairs. An error in the off-diagonals at layer 1 shows up as a diagonal error at layer 8. That compounding is the depth problem in one sentence.

---

## Part 03 — Scoring, compute model, rules

*The machinery is unusual and it shapes which algorithms can win.*

### 3.1 The score

```
MSE_final(m)  =  (1/n) · Σᵢ ( Ŷ_{L,i} − Y_{L,i} )²

s(m)  =  MSE_final(m) · max( 0.5 , C(m)/B(m) )

leaderboard  =  average of s(m) over the private suite      # LOWER IS BETTER
```

The compute factor rewards frugality, but the floor at 0.5 caps the bonus so that a very cheap, very inaccurate estimator cannot dominate. Practically: **you can never gain more than a factor of two by being cheap**, so if your method is accuracy-limited rather than budget-limited, spend the whole budget. All-layer MSE is reported alongside as a diagnostic but does not enter ranking.

### 3.2 The FLOP model — this is the interesting part

The contest is decided by an *analytical* FLOP budget, not wall-clock time, so faster hardware confers no advantage and the winner is decided by algorithm design. Accounting is done by `flopscope`, a NumPy-compatible drop-in: you import `flopscope.numpy` in place of `numpy` and every operation is tallied.

```
C(m)  =  F(m)  +  λ · R(m)

F(m)  # analytically counted FLOPs from fnp.* / flops.* calls
R(m)  # wall-clock time of anything NOT counted — plain numpy,
      #   Python scalar loops, uninstrumented libraries
λ     # an unfavourable conversion rate. Avoid paying this.
```

Three properties of flopscope that should shape your design:

- **Dispatch overhead is excluded.** Calling many small `fnp` ops does not penalise you, so you can write naturally structured code rather than fusing everything.
- **Symmetry is tracked.** Operations on symmetric arrays — covariance updates, Gram matrices — are charged at the cheaper symmetric-matrix cost rather than the full general cost. This is a deliberate nudge: the organisers expect winning methods to be covariance-based.
- **Escaping the instrumentation is punished, not rewarded.** Anything you do in raw numpy gets charged back through λ at a bad rate.

### 3.3 Failure handling

If a submission exceeds budget, raises, returns invalid shapes or non-finite values, exhausts memory, or trips an operational guard on a given MLP, the grader substitutes a **zero prediction** for that MLP and continues with the rest of the suite. No compute discount is applied to the fallback. Since the true activations average roughly 0.65 at this configuration, a zero prediction is catastrophic — it will contribute an MSE around 1.0 against a field competing near 10⁻⁷. **One crashed MLP destroys your entire run.** Defensive coding is worth more than a clever last 5%.

### 3.4 The private re-run — the rule that decides everything

The public leaderboard during Phases 1 and 2 is *not* the final ranking. After the Phase 2 deadline the grader re-executes each team's one designated final submission against a fresh, unseen MLP suite generated from a private seed never used during the open phases. Prize ranking comes exclusively from that re-run, and submissions that overfit to specific MLPs or seeds are penalised.

Statistically close finishes are first resolved by generating additional private MLPs to reduce uncertainty; remaining ties break on all-layer MSE and effective compute usage.

> **Design implication.** Build an estimator that is a function of the weights, not of the seed. Anything resembling a lookup table keyed on a specific network, or a hyperparameter tuned per-MLP on the public suite, is a trap. Test your method across widths and depths you were never scored on — the organisers have said they expect to change the (width, depth) setup in future rounds anyway.

### 3.5 Rules highlights

- Submissions are executable Python conforming to the starter-kit contract, not prediction files.
- All weights, lookup tables and precomputed artifacts must be bundled inside the tarball — there is no network access at evaluation time.
- Do not modify flopscope, read private seeds, or access grader-internal state. Grounds for disqualification.
- Every submission counts against the shared team budget regardless of who submitted it.
- Before the Phase 2 deadline each team designates **one** valid submission to carry into the final re-run.

> **The organisers' own warning about LLM-written code.** ARC states plainly that the FLOP-counting utility *is* hackable in ways that would be very unambiguously hacking once pointed out — such as modifying constants or counts held in memory. Contestants are responsible for ensuring their submissions do not hack it, regardless of whether or how they used an LLM. If you are generating code with a model, read what it emits around the accounting boundary. An automated reward-hack you did not notice is still disqualifying.

---

## Part 04 — Why ARC is running this

*The safety agenda that makes a Gaussian integral worth $100,000.*

### 4.1 The motivating question

ARC's long-run goal is to answer questions like *"are there unusual situations in which this system would undermine human control?"* Behavioural testing is the obvious approach and it is the wrong tool: a highly capable system need not fall for constructed honey-pots, and rare behaviours are precisely the ones sampling misses. Hence white-box methods that exploit access to the model's internals. ARC is explicit that they will ultimately use whichever method performs best — "white-box" is a hypothesis about what will win, not a constraint.

### 4.2 The induction analogy

ARC frames the research programme as a proof by induction:

- **Base case** — mechanistic estimation for networks at initialisation. This is the contest. Every network starts out randomly initialised, and these are the simplest networks.
- **Inductive step** — adapting those methods across each gradient step of training. ARC expects this to be much harder and has a vision but not a solution.

Why the step is hard, in their words: their methods start from Gaussian approximations to the activation distributions and then track the lowest-order deviations from Gaussianity. In a *trained* network, specific higher-order deviations become much more important. They suspect the fix involves auxiliary "advice" that points out which higher-order deviations matter and is updated incrementally with each gradient step — possibly in the form of a more structured weight distribution, i.e. an efficient compression of the model's weights.

### 4.3 The payoff if it works

**Catching deceptive alignment at train time.** If you could produce good mechanistic estimates for frontier models, you could detect deceptive alignment even if the model's behaviour looks benign on every training input.

**Mechanistic training.** Because these estimates are differentiable, you can do gradient descent on a mechanistic estimate instead of on sampled losses. ARC calls this mechanistic training, and demonstrated a proof of concept — *mechanistic distillation* between networks of different widths. It does not yet outperform ordinary training. The argument for why it might matter: SGD may never sample a rare high-loss dangerous event even once, producing a model that does not account for it at all; a mechanistic estimate could notice how the rare event might transpire and allocate capacity to avoiding it. Such models would generalise very differently — which is the whole point under distribution shift.

**Low-probability estimation.** This is where the advantage is largest. Monte Carlo is essentially useless for probabilities significantly below 1/N with N samples. Using a comparable FLOP count, ARC's method sometimes achieves relative error under 30% for probabilities *100 times lower* than 1/N. The mean-estimation task in this contest is a proxy; tail estimation is the actual prize.

### 4.4 Where it sits in ARC's wider agenda

The relevant thread runs: heuristic arguments and the presumption of independence (2022) → estimating tail risk in neural networks → competing with sampling → the "Matching Sampling Principle" (MSP), ARC's broader conjecture that a mechanistic explanation should be able to match what sampling tells you. Jacob Hilton has described the existence of an algorithm like the one this contest seeks as a special case of MSP — specifically the train-and-explain version — with evidence resting on examples where it appears to hold, absence of compelling counterexamples, and abstract philosophical reasoning. If you are writing for the $20k algorithmic-contribution prize, framing your method against MSP is likely to land better than framing it against the leaderboard.

---

## Part 05 — State of the art: cumulant propagation

*What the ARC paper does, why it works, and exactly where it fails.*

Wu, Lecomte, Winer, Robinson, Hilton & Christiano, *Estimating the expected output of wide random MLPs more efficiently than sampling*, arXiv:2605.05179 (2026). Code: `alignment-research-center/mlp_cumulant_propagation`.

### 5.1 The core idea

Rather than running samples, track an approximate representation of the *distribution* of activations at each layer, alternately passing it through linear maps and through the ReLU. Define pre-activations and activations:

```
Z⁽ˡ⁾ = W⁽ˡ⁾ X⁽ˡ⁻¹⁾        # linear step
X⁽ˡ⁾ = φ( Z⁽ˡ⁾ )          # nonlinear step
```

The justification for tracking *cumulants* specifically: W⁽ˡ⁾ is a large, unstructured matrix, so by the central limit theorem Z⁽ˡ⁾ is close to Gaussian *regardless* of the distribution of X⁽ˡ⁻¹⁾. Near-Gaussian distributions are well specified by their low-degree cumulants. So: track the low-degree cumulants through the layers.

This is made quantitative in the paper (Corollary S.2.14): the higher cumulants decay with width, with E[|κ_r|²] = O(n²⁻ʳ) for even r under stated index conditions. Cumulant order r costs you n²⁻ʳ — which is why truncating at low order is legitimate *at large width*, and why the method is fundamentally a width-asymptotic argument.

### 5.2 The specific tricks that matter

ARC is explicit that the high-level algorithm is not new — cumulant propagation appeared in Appendix D of *Formalizing the presumption of independence* (2022). What is new are details that turn out to be essential for low MSE:

- **Hermite expansions of the activation.** Expanding φ in the Hermite basis makes Gaussian expectations computable term by term.
- **Power cumulants.** Instead of Hermite-expanding φ, Hermite-expand the *powers* φ(z)ᵏ, and estimate power cumulants of the output. This lets you compute the Gaussian variance exactly while continuing to approximate the off-diagonal covariance.
- **Full trace for odd K.** When the maximum tracked cumulant order K is odd, additionally track a single scalar corresponding to the full trace of the (K+1)-order cumulant tensor — e.g. for K = 1, the trace of the covariance, Σᵢ Var[Xᵢ]. This captures a specific correlation that reduces error substantially at almost no cost.
- **Augmented variants** that track selected additional components of the (K+1)-order tensor (such as everything except the traceless part).

The `k_max` parameter in the reference implementation is the maximum cumulant order tracked in full. Section 6.4 of the paper ablates these choices — read that section before you design anything, because it tells you which ideas were already tried.

### 5.3 Results — and the depth cliff

| Claim | Detail |
|---|---|
| Asymptotic advantage | MC: MSE Θ(ε²) in time Θ(n²/ε²). Their algorithm: MSE O(ε²) in time O(n/ε²) — a factor of n faster (valid when ε² decays like 1/nᵏ, k ≥ 2) |
| Empirical (L = 4, n = 256) | Beats Monte Carlo across FLOP budgets spanning 7 orders of magnitude; in places reaches the same MSE with under 1/100 the FLOPs |
| Tails | Relative error under 30% for probabilities 100× below the 1/N Monte-Carlo floor |
| Wall-clock | Often *underperforms* MC in wall-clock, because no serious hardware optimisation was attempted (Appendix I) — precisely why the contest scores analytical FLOPs |
| **The gap** | **Guaranteed to beat MC at sufficiently large width for any fixed depth — but the dependence on depth is worse, and the methods break down as depth grows (Appendix D)** |

> **Read this if you read nothing else.** The headline experiments are at **L = 4**. The contest runs at **L = 8**. ARC says they are "very confident" their methods can be significantly improved, and the contest exists specifically to find that improvement. **Appendix D of the paper — the depth-scaling analysis — is the single most important thing to read.** It remains an open problem even to match sampling in other regimes, for example depth growing linearly in width.

---

## Part 06 — Measured baselines

*My own experiments at n = 256, L = 8. Reproduce these before trusting them.*

The following were measured directly: one random MLP at the contest configuration, ground truth from 2 × 10⁶ Monte-Carlo samples, all figures averaged over repeated trials where sampling is involved. Numbers will vary by a factor of a few across weight draws; treat them as orientation.

### 6.1 Structural constants

| Quantity | Value | Note |
|---|--:|---|
| Mean activation, final layer | ≈ 0.652 | He init preserves E[h²] ≈ 1 through depth |
| Per-neuron Var(h_L) | ≈ 0.207 | this sets the Monte-Carlo noise floor |
| Spread across neurons | σ ≈ 0.86 | heavy — neurons differ a lot; this is signal, not noise |
| FLOPs per forward sample | ≈ 1.05 × 10⁶ | 2n²L |
| MC samples affordable at budget | ≈ 32,400 | 3.4 × 10¹⁰ / 1.05 × 10⁶ |

### 6.2 What each method actually scores

| Method | Final-layer MSE | Cost vs budget | Comment |
|---|--:|--:|---|
| Zero prediction (the failure fallback) | ≈ 1.0 | — | what one crash costs you |
| Gaussian mean + full covariance propagation | 3.1 × 10⁻⁵ | < 1% | zero samples; correlation with truth 0.999985 |
| Plain Monte Carlo at full budget | 7.2 × 10⁻⁶ | 100% | matches the predicted Var/N = 6.4 × 10⁻⁶ |
| MC + Gaussian-linearisation control variate | 4.1 × 10⁻⁶ | ≈ 100% | a free 1.7× — cheapest real win available |
| **Warm-up round leader (reported)** | **≈ 9.0 × 10⁻⁸** | — | ≈ 75× below plain MC; config may have differed |

> **The three facts that fall out of this table.**
>
> **1. Compute is not the binding constraint.** Full covariance propagation costs under 1% of the budget and already lands within a factor of 4 of full-budget Monte Carlo. You have roughly two orders of magnitude of headroom to spend on higher-order corrections. Approximation quality, not FLOPs, is what separates competitors.
>
> **2. The Gaussian approximation is nearly right — and that is the problem.** Correlation with truth is 0.999985. Essentially all the remaining error is a small, structured, non-Gaussian residual. You are not fighting for the first three digits; you are fighting for the fifth onward.
>
> **3. Black-box variance reduction has a low ceiling.** The linear control variate gives 1.7×. Getting from 7 × 10⁻⁶ to 9 × 10⁻⁸ needs a 75× MSE reduction, i.e. ~8.7× in RMSE. Sampling-based routes would need ~5,600× more samples. That is not reachable inside the budget. **The frontier is mechanistic, exactly as ARC predicted.**

### 6.3 The cost of cumulants, tabulated

This table explains ARC's design choices better than any prose. At n = 256:

| Order | Object | Storage (entries) | Propagation cost / layer | Verdict |
|--:|---|--:|--:|---|
| 1 | mean vector | 2.6 × 10² | ~10⁵ | free |
| 2 | covariance matrix | 6.6 × 10⁴ | ~3 × 10⁷ | free (<1% of budget for all 8 layers) |
| 3 | third cumulant tensor | 1.7 × 10⁷ | ~4 × 10⁹ | **at the limit** — 8 layers ≈ the entire budget |
| 4 | fourth cumulant tensor | 4.3 × 10⁹ | ~10¹² | impossible — 34 GB just to store |

So the design space is sharply constrained: orders 1 and 2 in full, and then *partial, structured* information about orders 3 and 4 — traces, low-rank factors, diagonal blocks, sketches. This is precisely why ARC's paper tracks a single scalar trace of the (K+1)-order tensor rather than the tensor itself. Any competitive idea will be an answer to the question: **which O(n²)-sized summary of the third and fourth cumulants carries the most information about the final-layer means?**

---

## Part 07 — Where the difficulty actually is

*My reading of the open problem. Argue with it.*

### 7.1 Why depth breaks the method

Three mechanisms compound, and they are worth separating because they suggest different fixes.

**(a) Truncation error accumulates multiplicatively.** Each layer's output cumulants are computed from a *truncated* representation of the input distribution. That error becomes the input error for the next layer, where it is transformed again. Over 8 layers a per-layer relative error ε compounds toward something like (1+ε)⁸ − 1. Halving the per-layer error is worth roughly 8× at the output. This argues for spending budget on making a single layer's transfer step as exact as possible rather than on tracking more orders.

**(b) Non-Gaussianity is regenerated at every layer.** The CLT argument says Z⁽ˡ⁾ = W⁽ˡ⁾X⁽ˡ⁻¹⁾ is *pushed toward* Gaussianity by the linear step. But the ReLU immediately pushes back — it is exactly the operation that manufactures skewness. So the layer-to-layer dynamics are a competition between Gaussianising (linear) and de-Gaussianising (nonlinear) steps. At fixed width, deeper means more rounds of that competition, and the deviations reach a non-vanishing steady state rather than decaying.

**(c) The relevant expansion parameter is L/n, not 1/n.** This is the key connection to the wider literature. The finite-width perturbative theory of deep networks (Roberts, Yaida & Hanin) establishes that the depth-to-width ratio L/n governs when the perturbative expansion around the infinite-width Gaussian process breaks down. At n = 256 and L = 8, L/n = 1/32 — small but not negligible, and the corrections are O(L/n) at leading order with higher terms in (L/n)². ARC's method is, structurally, a truncated expansion in exactly this parameter. **Their "breaks down as depth grows" and the DLT literature's "L/n controls the expansion" are, I believe, the same statement.** That literature has already worked out the recursions for the leading finite-width corrections. Importing them is the most obvious under-exploited move available.

> **The strategic read.** ARC's paper is a width-asymptotic result imported into a fixed-width, moderate-depth regime. The statistical-physics deep-learning-theory community has spent five years on precisely the finite-width, finite-depth corrections that this regime needs — connected four-point correlators, the L/n expansion, critical initialisation. As far as I can tell those two literatures have not been fully joined. If you want an angle that is both likely to score and likely to win the algorithmic prize, that junction is where I would dig.

### 7.2 The mean is not zero, and this matters more than it looks

A trap I walked into while building this document: it is tempting to assume pre-activations are zero-mean and use the clean σ/√(2π) formula. They are not. ReLU outputs are non-negative, so E[X⁽ˡ⁾] ≠ 0, and therefore E[Z⁽ˡ⁺¹⁾] = W⁽ˡ⁺¹⁾E[X⁽ˡ⁾] ≠ 0 for every layer after the first. A covariance-only propagation that drops the mean scores MSE ≈ 1.0 — i.e. no better than predicting zero. You must carry the mean vector alongside the covariance, and the pairwise second-moment step needs the general non-zero-mean bivariate ReLU integral, which is materially harder than the clean arc-cosine kernel. Budget for that.

### 7.3 Where the remaining error concentrates

In my Gaussian-propagation run the residual was *not* a uniform constant: the systematic bias was 1.7 × 10⁻³, contributing only about 10% of the total MSE. The rest is neuron-specific. That means a global calibration constant buys you little — the correction you need is a function of each neuron's own (μ, σ, and higher structure). Any method whose correction term is constant across neurons is leaving most of the error on the table.

---

## Part 08 — An attack ladder

*A staged plan. Each rung is testable in isolation against the harness.*

| # | Rung | What it establishes |
|--:|---|---|
| 0 | Get the harness green end-to-end: `whest validate` → `--runner local` → `docker` → `package` | The contract, the environment, and the fact that you will not be zeroed out |
| 1 | Plain Monte Carlo at full budget | Your reference point: ≈ 7 × 10⁻⁶ |
| 2 | + antithetic pairs and Sobol / randomised QMC inputs | Cheap, robust variance reduction. Verify the gain is real at n = 256 — QMC degrades in high dimension |
| 3 | + Gaussian-linearisation control variate (Part 11) | Measured ≈ 1.7×. Confirms your plumbing for hybrid methods |
| 4 | Pure mean + full covariance propagation, no samples | ≈ 3 × 10⁻⁵ at <1% of budget. Your mechanistic skeleton |
| 5 | Port ARC's `kprop` and sweep `k_max`, power cumulants, the odd-K trace | Reproduces the published state of the art. Do not skip this — it tells you what is already tried |
| 6 | Hybrid: mechanistic estimate as the control variate for a small MC run | Strictly dominates both parents if the correlation is high, and it is. This is the obvious first original contribution |
| 7 | Structured third/fourth-order information: low-rank sketches, trace families, diagonal blocks | The real contest. See the cost table in 6.3 for what fits |
| 8 | Import finite-width DLT corrections — the O(L/n) connected-correlator recursions | The angle I would bet on. Also the most publishable |

### 8.1 Notes on specific rungs

**Rung 6 — why the hybrid is nearly free.** Let g(x) be the true final-layer activation vector and ĝ a cheap surrogate whose Gaussian expectation you know exactly. Then for any β,

```
E[g]  ≈  mean_i g(xᵢ)  −  β · ( mean_i ĝ(xᵢ) − E[ĝ] )
```

is unbiased for every β, and the optimal β is the per-neuron regression coefficient of g on ĝ. Since your mechanistic estimator is correlated with the truth at 0.999985, a surrogate built from it should strip most of the MC variance. The subtlety is that ĝ must be *evaluable on samples* and have a *known* expectation — the mechanistic estimate alone gives you the second but not the first. The linearised network gives you both.

**Rung 7 — the question to keep asking.** Which O(n²)-sized summary of the third and fourth cumulants carries the most information about the final-layer means? Candidates worth trying: the trace family ARC already uses; a rank-r factorisation of the third-cumulant tensor estimated by sketching; per-neuron marginal skewness and kurtosis only (n-sized, essentially free) fed into an Edgeworth correction; and pairwise-block information for the strongly correlated neuron pairs only.

**Cheap wins that are easy to forget.**

- Marginal skewness/kurtosis per neuron is O(n) to carry and gives you a per-neuron Edgeworth correction — directly addressing the finding in 7.3 that the residual is neuron-specific.
- The symmetric-matrix discount in flopscope means you should force symmetry explicitly (`(A + Aᵀ)/2`) so the accountant recognises it.
- Guard every MLP: wrap in try/except, check for NaN/Inf, check shapes, and keep a cheap fallback estimate ready. A zeroed MLP costs ≈ 1.0 against a field at 10⁻⁷.
- Budget headroom: leave margin under B. Going over is not a graceful degradation, it is a zero.

---

## Part 09 — The mathematical toolkit

*Tiered by what you actually need to write a competitive estimator.*

### Tier A — you cannot proceed without these

| Topic | Why you need it |
|---|---|
| Cumulants: joint cumulants, cumulant generating functions, moment ↔ cumulant conversion | The entire representation the method propagates |
| Wick's theorem / Isserlis' theorem | Computing Gaussian moments of products — every second-moment step |
| Hermite polynomials; Hermite expansion of ReLU and of ReLUᵏ | ARC's mechanism for pushing distributions through the nonlinearity |
| Stein's lemma / Gaussian integration by parts | Relates E[x·f(x)] to E[f′(x)]; underlies the linearisation and control variates |
| Edgeworth and Gram–Charlier expansions | The formal series in cumulants around a Gaussian — literally what truncation means |
| Arc-cosine kernel (Cho & Saul 2009) | Closed-form E[ReLU(u)ReLU(v)] for correlated zero-mean Gaussians |
| CLT with Berry–Esseen rates | Justifies why pre-activations are near-Gaussian and how near |
| Numerical linear algebra: Cholesky, eigendecomposition, randomised SVD, low-rank updates | Everything is an n × n covariance operation under a FLOP budget |

### Tier B — the deep-network theory this problem is secretly about

| Topic | Canonical sources |
|---|---|
| Neural Network Gaussian Process (NNGP): the infinite-width covariance recursion K⁽ˡ⁺¹⁾ = f(K⁽ˡ⁾) | Neal 1996; Lee et al. 2018; Matthews et al. 2018 |
| Mean-field signal propagation, order/chaos transition, edge of chaos | Poole et al. 2016 ("Exponential expressivity"); Schoenholz et al. 2017 ("Deep Information Propagation") |
| **Finite-width perturbation theory: the 1/n expansion, connected correlators, the L/n parameter** | **Roberts, Yaida & Hanin, *The Principles of Deep Learning Theory* (2022) — free at deeplearningtheory.com** |
| Non-Gaussian processes at finite width | Yaida, "Non-Gaussian processes and neural networks at finite widths" |
| Products of many random matrices; log-normal activation norms at depth | Hanin; Hanin & Nica |
| Neural Tangent Kernel | Jacot et al. 2018 — adjacent, mostly relevant if you extend to training |
| Random matrix theory basics; free probability | Background for spectral treatment of W Σ Wᵀ |

### Tier C — variance reduction, for the hybrid route

| Technique | Applicability here |
|---|---|
| Control variates | Strongest option; measured 1.7× with the crudest possible surrogate |
| Antithetic variates | Nearly free given the symmetric input distribution |
| Quasi-Monte Carlo (Sobol, randomised QMC) | Effectiveness degrades in n = 256 dimensions — test, do not assume |
| Rao–Blackwellisation | Analytically integrate out whatever you can condition on |
| Importance sampling | Marginal for mean estimation; essential if the task ever moves to tails |
| Common random numbers | For fair A/B comparison of your own variants during development |

### Suggested reading order

1. ARC blog post *Mechanistic estimation for wide random MLPs* — 20 minutes, gives you the shape.
2. The starter kit and flopscope source — until you can run `whest run --runner local`.
3. The paper's main body, then **Section 6.4 (ablations)** and **Appendix D (depth scaling)**.
4. The `mlp_cumulant_propagation` repo — run it, sweep `k_max`, watch what breaks at L = 8.
5. *The Principles of Deep Learning Theory*, chapters on the 1/n expansion and L/n.
6. *Competing with sampling* and *Estimating tail risk in neural networks* — for the write-up prize.

---

## Part 10 — Annotated resource index

*Everything worth opening, grouped and rated. **[core]** = read it; **[high]** = read it soon; **[context]** = when you have time.*

### 10.1 The contest itself

**Challenge page — ARC White-Box Estimation Challenge 2026** — **[core]**
<https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026>
The authoritative statement of the task, the compute model, scoring, prizes and timeline. Tabs for Leaderboard, Discussion, Insights, Resources, Rules.

**Starter kit — AIcrowd/whest-starterkit** — **[core]**
<https://github.com/AIcrowd/whest-starterkit>
The six-stage ladder: local iteration → contract validation → local runner → subprocess runner → Docker runner → packaging. Your estimator.py must conform to the contract published here, not to the paper.

**flopscope — the FLOP accounting library** — **[core]**
<https://github.com/AIcrowd/flopscope>
NumPy-compatible drop-in that counts every floating-point operation. Read the source: knowing exactly what is charged, what the symmetry discount covers, and where the uninstrumented boundary sits is worth real leaderboard positions.

**WhestBench Explorer — interactive visualisation** — **[high]**
<https://aicrowd.github.io/whestbench-explorer/>
Generate random MLPs at chosen (width, depth) and view per-neuron Monte-Carlo ground truth as a heatmap. Good for building intuition about how the target varies across neurons and layers.

**Official rules page** — **[core]**
<https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules>
Submission caps per phase, team budget sharing, the designation of a final submission, disqualification conditions.

**Discussion forum · Discord · email** — **[high]**
aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/discussion · discord.gg/4gyQvzWPJ · arc-whestbench@aicrowd.com
Forum for task and rules questions; GitHub Issues for bugs in flopscope or the harness; email for administrative matters.

**Manifold market on the winning score** — **[context]**
<https://manifold.markets/Loppukilpailija/what-will-be-winning-score-in-arcs>
Crowd forecast of the winning score. Cited the warm-up leader at an adjusted score of 9.03e-8, which is the most useful public calibration point available.

### 10.2 The paper and its code

**Estimating the expected output of wide random MLPs more efficiently than sampling** — **[core]**
<https://arxiv.org/abs/2605.05179>
Wu, Lecomte, Winer, Robinson, Hilton, Christiano (2026). The technical foundation of the entire contest. Priority sections: 6.3 width scaling, 6.4 ablations, 6.6 low-probability estimation, Appendix D depth scaling, Appendix I wall-clock, Supplement S.1–S.2 the cumulant machinery.

**mlp_cumulant_propagation — official implementation** — **[core]**
<https://github.com/alignment-research-center/mlp_cumulant_propagation>
The 'kprop' algorithm plus experiment and plotting scripts. k_max sets the maximum cumulant order tracked in full. Running this at L = 8 and watching where it degrades is probably the single highest-value hour you can spend.

**Mechanistic estimation for wide random MLPs — blog version** — **[core]**
<https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/>
Hilton, 7 May 2026. The readable summary: results, significance, the induction analogy, and an honest account of why the method does not yet extend to trained networks. Start here.

### 10.3 ARC's agenda — context for the write-up prize

**Announcing the ARC White-Box Estimation Challenge** — **[core]**
<https://www.alignment.org/blog/announcing-the-arc-white-box-estimation-challenge/>
Hilton, Christiano & Wu, 2 June 2026. The organisers' own framing, the LLM-use policy, and the footnote about the omitted final linear layer.

**LessWrong / Alignment Forum discussion thread** — **[high]**
<https://www.lesswrong.com/posts/Kben8CzS4awCwNw5c/announcing-the-arc-white-box-estimation-challenge>
The comment thread contains organiser replies clarifying what counts as a real solution, the relationship to the Matching Sampling Principle, and interest in extending to trained MLPs. Worth reading in full before writing a technical report.

**Competing with sampling** — **[high]**
<https://www.alignment.org/blog/competing-with-sampling/>
Defines what makes an algorithm 'mechanistic' and sets the research target the paper is aimed at. Also contains the 'compression as a possible MSP approach' section. This is the framing document.

**Estimating tail risk in neural networks** — **[high]**
<https://www.alignment.org/blog/estimating-tail-risk-in-neural-networks/>
The safety case: deliberate subversion of adversarial training, and why catching deceptive alignment at train time is the goal.

**Formalizing the presumption of independence** — **[high]**
<https://arxiv.org/abs/2211.06738>
Christiano et al., 2022. Appendix D introduced cumulant propagation. Historical root of the entire method.

**Low probability estimation in language models** — **[high]**
<https://www.alignment.org/blog/low-probability-estimation-in-language-models/>
Where the mechanistic advantage is largest and where the agenda is actually heading. Monte Carlo is useless below 1/N; mechanistic estimates are not.

**AlgZoo: uninterpreted models with fewer than 1,500 parameters** — **[context]**
<https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/>
Demonstrates how hard white-box estimation is for *trained* networks, even tiny ones. Useful calibration for how much the random-init setting is a simplification.

**ARC main site — for the Matching Sampling Principle** — **[high]**
<https://www.alignment.org/>
ARC's agenda has moved substantially in the last 15 months around MSP. Check the blog index for the newest posts; MSP framing is likely to matter for the algorithmic-contribution prize.

### 10.4 Background literature

**The Principles of Deep Learning Theory — Roberts, Yaida & Hanin** — **[core]**
<https://deeplearningtheory.com>
Free full text. The systematic 1/n perturbative expansion of finite-width networks: connected correlators, the four-point vertex, and the depth-to-width ratio L/n as the governing parameter. If you read one background text, read this. Its central result is, I believe, the same phenomenon as ARC's depth breakdown.

**Deep Information Propagation — Schoenholz, Gilmer, Ganguli, Sohl-Dickstein** — **[high]**
<https://arxiv.org/abs/1611.01232>
Mean-field signal propagation, the order-to-chaos transition, and why criticality matters. He init at variance 2/n sits at the critical point, which is exactly why depth degrades estimates slowly rather than catastrophically.

**Exponential expressivity in deep neural networks through transient chaos — Poole et al.** — **[high]**
<https://arxiv.org/abs/1606.05340>
The original mean-field treatment of how correlations between inputs evolve with depth.

**Deep Neural Networks as Gaussian Processes — Lee et al.** — **[high]**
<https://arxiv.org/abs/1711.00165>
The NNGP covariance recursion in its cleanest form. Your zeroth-order mechanistic estimator is this recursion applied to a single fixed weight draw.

**Kernel Methods for Deep Learning — Cho & Saul (2009)** — **[high]**
<https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning>
Source of the arc-cosine kernel: the closed form for E[ReLU(u)ReLU(v)] under a correlated zero-mean bivariate Gaussian. You will use this constantly.

**Non-Gaussian processes and neural networks at finite widths — Yaida** — **[context]**
<https://arxiv.org/abs/1910.00019>
Finite-width corrections to the NNGP picture, written from the statistical-physics side.

**Products of many large random matrices and gradients in deep neural networks — Hanin & Nica** — **[context]**
<https://arxiv.org/abs/1812.05994>
Why quantities in deep networks become log-normally distributed with depth. Directly relevant to why variance estimates degrade at L = 8.

**Neural Tangent Kernel — Jacot, Gabriel & Hongler** — **[context]**
<https://arxiv.org/abs/1806.07572>
Adjacent rather than central: matters if you pursue the 'inductive step' of tracking estimates through training.

**Delta Method / Edgeworth expansion — any graduate asymptotic statistics text** — **[context]**
Hall, 'The Bootstrap and Edgeworth Expansion' (1992) is the standard reference.
For the formal theory of expanding an expectation in cumulants around a Gaussian, including when the series is asymptotic rather than convergent — which it is here.

---

## Part 11 — Formula cheat sheet

*Everything below was verified numerically against 4 × 10⁶ Monte-Carlo samples.*

### 11.1 First moment of a ReLU under a Gaussian

For Z ~ N(μ, σ²), with Φ and φ the standard normal CDF and PDF:

```
E[ ReLU(Z) ]  =  μ·Φ(μ/σ)  +  σ·φ(μ/σ)

# zero-mean special case:
E[ ReLU(Z) ]  =  σ / √(2π)   ≈  0.3989 σ        when μ = 0
```

*Verified: μ=0.3, σ=1.4 → analytic 0.72129, empirical 0.72042. Remember that μ ≠ 0 for every layer after the first — see 7.2.*

### 11.2 Second moment: the arc-cosine kernel

For (u, v) jointly Gaussian, zero mean, standard deviations σᵤ, σᵥ, correlation ρ, and θ = arccos(ρ):

```
E[ ReLU(u) · ReLU(v) ]  =  (σᵤ σᵥ / 2π) · ( sin θ  +  (π − θ)·cos θ )
```

*Verified: σᵤ=1.3, σᵥ=0.7, ρ=−0.4 → analytic 0.06558, empirical 0.06578. For the non-zero-mean case there is no comparably clean form; use a bivariate Gauss–Hermite quadrature (32 nodes per axis was ample in my tests) or the Rosenbaum/Kan expressions in terms of the bivariate normal CDF.*

### 11.3 The layer recursion, in full

```
m⁽⁰⁾ = 0 ,  Σ⁽⁰⁾ = Iₙ

for ℓ = 1 … L:
    μ_z  =  W⁽ˡ⁾ m⁽ˡ⁻¹⁾                       # pre-activation mean
    Σ_z  =  W⁽ˡ⁾ Σ⁽ˡ⁻¹⁾ W⁽ˡ⁾ᵀ                  # pre-activation covariance (symmetric!)
    σ    =  sqrt( diag(Σ_z) )
    m⁽ˡ⁾ =  μ_z·Φ(μ_z/σ) + σ·φ(μ_z/σ)          # THIS ROW IS YOUR ANSWER at ℓ = L
    M₂   =  E[ReLU(zᵢ)ReLU(zⱼ)]                # pairwise, via 11.2 or quadrature
    Σ⁽ˡ⁾ =  M₂ − m⁽ˡ⁾ (m⁽ˡ⁾)ᵀ
```

Force `Σ_z = (Σ_z + Σ_zᵀ)/2` explicitly so flopscope's symmetry tracking applies the cheaper cost. Measured performance of exactly this recursion at n=256, L=8: MSE ≈ 3.1 × 10⁻⁵ using under 1% of the budget.

### 11.4 Edgeworth correction structure

For a near-Gaussian Z with cumulants κ₃, κ₄ beyond the Gaussian part, the formal expansion of any smooth-enough functional is

```
E[f(Z)]  ≈  E_G[f]  +  (κ₃/3!)·E_G[f⁽³⁾]  +  (κ₄/4!)·E_G[f⁽⁴⁾]  +  (κ₃²/(2·(3!)²))·E_G[f⁽⁶⁾]  + …
```

For f = ReLU the derivatives are distributions: ReLU″ = δ, ReLU‴ = δ′, ReLU⁗ = δ″, and E[δ⁽ᵏ⁾(Z)] = (−1)ᵏ p⁽ᵏ⁾(0) where p is the density of Z. A consequence worth noting: for a *zero-mean* Z the κ₃ term vanishes identically (p′(0) = 0), so the leading non-Gaussian correction to E[ReLU] comes from the fourth cumulant. For non-zero mean it does not vanish. Derive both cases yourself before relying on them.

### 11.5 Control variate

```
Ê  =  mean_i g(xᵢ)  −  β ⊙ ( mean_i ĝ(xᵢ) − E[ĝ] )

# unbiased for any β; optimal β is the per-neuron regression of g on ĝ
β  =  Cov(g, ĝ) / Var(ĝ)          # estimated from the same samples

# cheapest useful surrogate: the Gaussian-equivalent linearisation
A  =  Π_ℓ  D⁽ˡ⁾ W⁽ˡ⁾ ,   D⁽ˡ⁾ = diag( Φ(μ_z⁽ˡ⁾ / σ⁽ˡ⁾) ) ,   ĝ(x) = A x ,   E[ĝ] = 0
```

*Measured: plain MC 6.8 × 10⁻⁶ → with this control variate 4.1 × 10⁻⁶, a 1.7× improvement for essentially no extra FLOPs. A better surrogate should do considerably better.*

### 11.6 Useful constants at this configuration

```
E[h²] ≈ 1 at every layer            # He init preserves the second moment
Var(Z⁽ˡ⁾) ≈ 2                       # pre-activation variance
E[h] ≈ 0.652 ,  Var(h) ≈ 0.207      # measured, final layer, n=256 L=8
FLOPs per MC sample  = 2n²L ≈ 1.05 × 10⁶
samples affordable   ≈ 32,400
MC noise floor       ≈ Var(h)/N ≈ 6.4 × 10⁻⁶
```

---

## Part 12 — Timeline, glossary, contacts

### 12.1 Timeline

| Milestone | Date | Note |
|---|---|---|
| Warm-up round opens | 28 May 2026 | Scores do not carry forward |
| Phase 1 — open competition | 18 June – 31 July 2026 | Public leaderboard, not final ranking |
| Phase 2 — final submission | 1 August – 19 September 2026 | **This is where the prizes are decided** |
| Final evaluation & due diligence | 20 – 30 September 2026 | Private re-run on unseen seed |
| Results announced | 1 October 2026 | |
| NeurIPS Competition Track workshop | 11 – 12 December 2026 | TBD |

*After Phase 2 starts, only operational fixes applied consistently to all submissions are allowed. A post-competition results report covering the final leaderboard, baseline comparisons and a taxonomy of submitted methods will be published by the organisers.*

### 12.2 Glossary

| Term | Meaning |
|---|---|
| **White-box estimation** | Predicting a model's behaviour by reading its weights, rather than by observing its outputs on inputs |
| **Mechanistic estimate** | An estimate produced without running the model on any input at all — the strongest form of white-box |
| **Cumulant propagation ("kprop")** | Tracking low-order cumulants of the activation distribution layer by layer instead of sampling |
| **k_max** | Maximum cumulant order tracked in full by ARC's implementation |
| **Power cumulants** | Hermite-expanding φ(z)ᵏ rather than φ(z), to get the Gaussian variance exactly |
| **Hermite expansion** | Writing a function in the orthogonal polynomial basis for the Gaussian measure, making Gaussian expectations term-by-term computable |
| **NNGP** | Neural Network Gaussian Process — the infinite-width limit in which activations are exactly Gaussian and the covariance obeys a closed recursion |
| **He initialisation** | Weights drawn N(0, 2/n); the variance choice that keeps E[h²] constant through ReLU layers |
| **flopscope** | The contest's analytical FLOP accounting library; a NumPy drop-in |
| **λ (lambda)** | The unfavourable rate at which uncounted wall-clock time is charged back against your FLOP budget |
| **MSP** | Matching Sampling Principle — ARC's broader conjecture that mechanistic explanation should match what sampling reveals |
| **Deceptive alignment** | A model that behaves well on all training inputs while retaining behaviour that would undermine control elsewhere — the failure mode this agenda targets |
| **L/n** | Depth-to-width ratio; the parameter governing breakdown of perturbative expansions around the infinite-width limit. Here 8/256 = 1/32 |

### 12.3 Contacts

| Purpose | Where |
|---|---|
| Task / rules questions | AIcrowd discussion forum on the challenge page |
| Bugs in tooling | GitHub Issues on AIcrowd/whest-starterkit or AIcrowd/flopscope |
| Community | discord.gg/4gyQvzWPJ |
| Administrative | arc-whestbench@aicrowd.com |

### 12.4 Citation

```
Wilson Wu, Victor Lecomte, Michael Winer, George Robinson, Jacob Hilton
and Paul Christiano. "Estimating the expected output of wide random MLPs
more efficiently than sampling." arXiv:2605.05179, 2026.
```

---

> **Closing note.** The distance between where the published state of the art sits and where the warm-up leaderboard already sits suggests the field moved quickly once the problem was made concrete and scored. Two things follow. First, read the discussion forum and the leaderboard before investing weeks in an idea — someone may have already tried it. Second, ARC has said outright that they are very confident their methods can be significantly improved, and that they are not sure whether generalisable insights can be drawn from strong submissions that are primarily LLM-written. A clearly explained method that a human understands is worth $20,000 independent of where it places.
