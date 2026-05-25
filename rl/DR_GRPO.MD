# Note on GRPO Done Right (DR. GRPO)

- Most base models already exhibit the self-reflection abilities (e.g., Aha moments) during pre-training (The hypothesis is "those base models are pre-trained with concatenated question-answers")
- RL helps to amplify those self-reflection behaviors and the correct responses also increase after RL-tuned
- The response length also substantially increase in both correct/incorrect answers 
- The average length of incorrect responses are notably longer than correct ones (perhaps due to hard questions require longer reasoning chains?)
- Self-reflection does not correlate to the increased accuracy during inference (Though the self-reflection can improve exploration during training) => this behavior is not clean and accurate to explain the improved accuracy after RL.

## 1. Basics of RL for Language Models
Language model generation can be formulated as a **token-level Markov Decision Process (MDP)**.

An MDP is written as:

```math
\mathcal{M} = (\mathcal{S}, \mathcal{A}, r, p_Q)
```

In the LLM setting:

- `S` = state space
- `A` = action space
- `r` = reward function
- `p_Q` = distribution over input questions/prompts

---

## 1.1 State

At generation step `t`, the state is the concatenation of:

1. the input question/prompt `q`
2. the output tokens generated so far `o_<t`

$$
s_t = q ; o_{\lt t}
$$

or explicitly:

```math
s_t = [q_1, ..., q_M, o_1, ..., o_{t-1}]
```

Intuition:

> The state is the full current context of the autoregressive generation process.

Example:

```text
Question: Solve 2x + 1 = 5
Generated so far: We subtract 1 from both sides
```

The current state is:

```text
Solve 2x + 1 = 5 ; We subtract 1 from both sides
```

---

## 1.2. Action

At each step, the policy model selects the next token from the vocabulary: $o_t \in \mathcal{A}$.

The policy is:

```math
\pi_\theta(\cdot | s_t)
```

This means:

> Given the current context `s_t`, the LLM outputs a probability distribution over possible next tokens.

Then one token is selected, usually by sampling during RL training.

---

## 1.3. Deterministic transition

After the next token `o_t` is selected, the next state is obtained by appending that token to the current context:

```math
s_{t+1} = s_t ; [o_t]
```

This transition is **deterministic** because once the token is chosen, the next state is fixed.

Important distinction:

- The **policy** can be stochastic because it samples the next token.
- The **transition** is deterministic because the chosen token is simply appended to the context.

---

## 1.4. Initial state

The generation process starts by sampling an initial state from a set of questions:

```math
s_1 = q \sim p_Q
```

In simple terms:

> The prompt/question is the initial state of the episode.

---

## 1.5. End of trajectory

The trajectory ends when either: (1) the model generates the `[eos]` token, or (2) the model exhausts the generation budget.

A full trajectory is the full generated response: $[q;o]$, where $o = [o_1, o_2, ..., o_T]$

---

## 1.6. RL Objective in LLM Post-Training

The key idea of RL for LLM post-training is to maximize the entropy/KL-regularized objective ([Schulman et al., 2017a](#ref-schulman2017a)), which encourages the policy to generate the high-reward responses while staying close to reference model:

$$
J(\pi_\theta)= \mathbb{E}_{q \sim p_Q} \left[\mathbb{E}_{o \sim \pi_\theta(\cdot \mid q)} \left[R(q,o) \right] - \beta D_{\mathrm{KL}} \left[ \pi_\theta(\cdot \mid q) \,\|\ \pi_{\mathrm{ref}}(\cdot \mid q) \right] \right]
$$

Before considering this objective for Language Model Post-Training, let's look back to its fundamental lens (Traditional RL => Policy Gradient => Entropy-Regularized RL). 

In basic RL, the objective function is defined by: 

$$
J(\pi_\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
R(\tau)
\right]
$$

where $\tau$ is a trajectory sampled from the policy $\pi_\theta$, and $R(\tau)$ is the total return of that trajectory. 

In policy gradient, we update the policy parameters $\theta$ in the direction that increases the expected return:

$$
\nabla_\theta J(\pi_\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
[
\nabla_\theta \log \pi_\theta(\tau) R(\tau)
]
$$

In simple terms:

> - In traditional RL lens, just increase the probability of trajectories that have high total reward
> - In LLM post-training lens, as $\log \pi_\theta(\tau) = \sum_t \log \pi_\theta(o_t | q, o_{<t})$, the gradient scales the log-probability of every individual token in high-reward responses.

However, maximizing return alone can make the policy collapse too early into deterministic actions. `Entropy-regularized RL` reframes this by bridging the `Maximum Entropy Principle` from information theory to policy optimization. Under this view, **the policy does not only maximize reward but also being regularized by an information/certainty cost**, which discourages the policy from becoming more confident than the reward evidence justifies. Here, let's revisit the basic concepts of Maximum Entropy Principle.

### Maximum Entropy Principle

The maximum entropy principle comes from information theory. It says that *the most unbiased, logically consistent probability distribution to represent a system is the one that maximizes Shannon entropy, subject to whatever partial information or constraints are already known*

In simple terms, maximum entropy means:

> Do not assume more than what the given evidence or constraints justify.

Entropy measures how uncertain or uncommitted a distribution is:

$$
H(p) = -\sum_x p(x)\log p(x).
$$

A high-entropy distribution spreads probability mass more broadly, while a low-entropy distribution is more confident or concentrated.

Different constraints lead to different maximum-entropy distributions.

If the only constraint is that probabilities must sum to one,

$$
\sum_x p(x)=1,
$$

then the maximum-entropy distribution is uniform:

$$
p(x)=\frac{1}{|\mathcal X|}.
$$

>This represents the case where we have no prior knowledge, so every outcome is treated equally.

If we also know an expected value constraint, such as

$$
\mathbb{E}_{p}[f(x)] = c,
$$

then the maximum-entropy solution takes an exponential or Boltzmann form:

$$
p^*(x) \propto \exp(\lambda f(x)).
$$

>This means the distribution is still as uncertain as possible, but now it must respect the evidence encoded by \(f(x)\).

If we already have a prior distribution $p_0(x)$, the principle becomes Minimizing Relative Entropy (also known as KL divergence). Instead of choosing the distribution closest to uniform, we choose the distribution closest to the prior while satisfying the known constraints.

For example, if the added constraint is an expected value condition: $ \mathbb{E}_{p}[f(x)] = c,$

then the solution has the form:

$$
p^*(x) \propto p_0(x)\exp(\lambda f(x)).
$$

>In simple terms, the prior is not discarded; it is only tilted by the new constraint.

Therefore, the general idea is:

> maximum entropy determines the least committed distribution allowed by the constraints; adding more knowledge changes the form of the distribution.

Entropy-regularized RL modifies the objective by adding an entropy bonus:

$$
J_{\text{ent}}(\pi_\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_t \gamma^t
\left(
r(s_t,a_t)
+
\alpha H(\pi_\theta(\cdot|s_t))
\right)
\right].
$$

The entropy term encourages the policy to remain stochastic, which helps exploration and prevents premature collapse.

From an information-theoretic perspective, entropy regularization can be viewed as a special case of KL regularization. Since

$$
H(\pi(\cdot|s))
=
\log |\mathcal{A}|
-
D_{\mathrm{KL}}
\left(
\pi(\cdot|s)
\| U(\cdot|s)
\right),
$$

maximizing entropy is equivalent to minimizing KL divergence from a uniform action prior, up to a constant.

Therefore, entropy regularization can be interpreted as KL regularization toward a uniform prior.

A natural generalization is to replace the uniform prior with an informative prior policy \(\pi_0\):

$$
J_{\text{KL}}(\pi_\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_t \gamma^t
\left(
r(s_t,a_t)
-
\beta
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot|s_t)
\|
\pi_0(\cdot|s_t)
\right)
\right)
\right].
$$

Here, \(\pi_0\) can represent prior knowledge, such as an expert policy, a behavior policy, the previous policy, or a pretrained model.

Thus, entropy-regularized RL is the special case where the prior is uniform, while KL-regularized RL allows the prior to be informative.

## References

- <a id="ref-schulman2017a"></a>
John Schulman, Xi Chen, and Pieter Abbeel. *Equivalence Between Policy Gradients and Soft Q-Learning*. arXiv preprint arXiv:1704.06440. https://arxiv.org/abs/1704.06440, 2017a.