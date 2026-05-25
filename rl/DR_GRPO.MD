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

Before considering this objective for Language Model Post-Training, let's look back to its fundamental lens (Traditional RL => Policy Gradient => Information Theory => Entropy/KL-Regularized RL). 

In basic RL, the objective function is defined by: 

$$
J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$

where $\tau$ is a trajectory sampled from the policy $\pi_\theta$, and $R(\tau)$ is the total return of that trajectory. 

In policy gradient, we update the policy parameters $\theta$ in the direction that increases the expected return:

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [ \nabla_\theta \log \pi_\theta(\tau) R(\tau)]
$$

In simple terms:

> - In traditional RL lens, just increase the probability of trajectories that have high total reward
> - In LLM post-training lens, as $\log \pi_\theta(\tau) = \sum_t \log \pi_\theta(o_t | q, o_{<t})$, the gradient scales the log-probability of every individual token in high-reward responses.

However, maximizing return alone can make the policy collapse too early into deterministic actions. `Entropy/KL-regularized RL` reframes this by bridging the `Maximum Entropy Principle` from information theory to policy optimization. Under this view, **the policy does not only maximize reward but also being regularized by an information/certainty cost**, which discourages the policy from becoming more confident than the reward evidence justifies. Here, let's revisit the basic concepts of Maximum Entropy Principle.

### Maximum Entropy Principle (MaxEnt)

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

For example, if the added constraint is an expected value condition: $\mathbb{E}_{p}[f(x)] = c$,

then the solution has the form:

$$
p^*(x) \propto p_0(x)\exp(\lambda f(x)).
$$

>In simple terms, the prior is not discarded; it is only tilted by the new constraint.

Therefore, the general idea is:

> maximum entropy determines the least committed distribution allowed by the constraints; adding more knowledge (more constraints) changes the form of the distribution.

### Conceptual Lens of Deep RL via MaxEnt

In RL, the agent makes a sequential decisions over time (a trajectory $\tau$). By applying the Maximum Entropy Principle to the distribution of these trajectories, the abstract information-theoretic limits directly dictate the design of modern DRL algorithms. Depending on what we use as the Prior Base, the MaxEnt principle naturally splits DRL into two conceptual paradigms. 

> Notes: *In practical DRL, because the settings also depend on KL direction, state-wise vs trajectory-level objective, discrete vs continuous actions, and objective implementations, this MaxEnt-based distinction is not the strict taxonomy of DRL.*

#### A. The Uniform Prior Paradigm: Entropy-Regularized RL (Soft RL)
When an agent enters an unknown environment with no pre-existing skills, its default state of knowledge is a Uniform Prior.

To maximize the trajectory entropy subject to the expected reward constraint, we flip the Boltzmann Lagrange formulation into an unconstrained objective function:

$$
J_{\text{ent}}(\pi_\theta)=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_t \gamma^t \left(r(s_t,a_t) + \alpha \mathcal{H}(\pi_\theta(\cdot|s_t)) \right)\right]
$$

Here, the Shannon Entropy $\mathcal{H}$ acts as a cost for overconfidence. Based on the MaxEnt principle, because we only constrain the expected rewards, the optimal policy is mathematically forced to take the Boltzmann (Softmax) form:

$$
\pi^* (a|s) \propto \exp\left(\frac{Q^*(s,a)}{\alpha}\right)
$$

- In DRL, SAC is the pure, deep realization of this principle. It modifies the foundational Bellman Equation into the Soft Bellman Equation, explicitly training the network to converge onto the exact Boltzmann distribution derived above.

- In policy gradient methods like standard PPO, this term is often added as a shallow heuristic (entropy bonus) simply to inject noise and patch the exploration problem.

#### B. The Informative Prior Paradigm: KL-Regularized RL

When we do not start from scratch, but already possess a valuable baseline policy, maximizing entropy toward a uniform distribution becomes counterproductive. Instead, we transition to Minimizing Relative Entropy (KL Divergence) toward an informative prior policy ($\pi_0$).By substituting the uniform base with $\pi_0$, the objective transforms into:

$$J_{\text{KL}}(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \gamma^t r(s_t, a_t) \right] - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_0)$$

Following a sketchy view on Exponential Tilting of MaxEnt we derived earlier, the optimal policy is naturally characterized by the prior baseline tilted by the reward signal: 

$$\pi^* \propto \pi_0 \exp(\cdot)$$

This paradigm anchors two major milestones in RL based on how $\pi_0$ is defined:

- Moving Prior (TRPO / PPO-family): Here, $\pi_0 = \pi_{\text{old}}$ (the policy from the previous training iteration). The constraint is optimization stability. The algorithm forces the new policy to stay close to its immediate past self to prevent catastrophic gradient collapses during training.

- Frozen Prior (LLM Post-Training / PPO-HF / DPO): Here, $\pi_0 = \pi_{\text{ref}}$ (the frozen Supervised Fine-Tuned model). The constraint is knowledge preservation. The algorithm forces the LLM to optimize for human preferences (rewards) but strictly penalizes it if it drifts away from its foundational linguistic and factual capabilities ($\pi_{\text{ref}}$).

> Notes: *Entropy maximization can be interpreted as minimizing KL divergence to a uniform prior, while KL regularization generalizes this idea to an informative prior such as an old policy or a frozen reference model. $H(\pi) = \log |A| - D_{\text{KL}}(\pi \parallel U)$*

### RL in LLM Post-Training through MaxEnt Lens

The key idea of RL for LLM post-training is to maximize the entropy/KL-regularized objective ([Schulman et al., 2017a](#ref-schulman2017a)), which encourages the policy to generate the high-reward responses while staying close to reference model:

$$
J(\pi_\theta)= \mathbb{E}_{q \sim p_Q} \left[\mathbb{E}_{o \sim \pi_\theta(\cdot \mid q)} \left[R(q,o) \right] - \beta D_{\mathrm{KL}} \left[ \pi_\theta(\cdot \mid q) \,\|\ \pi_{\mathrm{ref}}(\cdot \mid q) \right] \right]
$$

After revisiting some fundamental concepts, we can see it as an unconstrained Lagrangian dual form of the Maximum Entropy principle operating under an Informative Prior. This objective is usually represents RLHF, in which:

- The Informative Prior ($\pi_{\text{ref}}$): Instead of assuming a flat action space (Uniform Prior), the system anchors itself to the SFT model. This constrains the optimization manifold strictly within the probability space of pre-existing linguistic capability and factual knowledge.

- The Constraint Feature ($ R(q,o) $): The Reward Model acts exactly as the feature function $f(x)$ in the expected value constraint ($\mathbb{E}_{\pi}[f(x)] = c$). It dictates the directional vector along which the prior distribution will be deformed.

- The Dual Coefficient / Inverse Multiplier ($\beta$): The KL weight defines the exact exchange rate between reward maximization and information divergence. In the pure MaxEnt primal problem (minimizing KL subject to a reward target), $\beta$ represents the inverse Lagrange multiplier ($\beta = 1/\lambda$).

Because this objective matches the structure of the mathematical dual of a MaxEnt problem, the policy leads to the exact Exponential Tilting closed-form solution:

$$
\pi^* (o \mid q) = \frac{1}{Z(q)} \pi_{\text{ref}}(o \mid q) \exp \left( \frac{R(q, o)}{\beta} \right)
$$

**Practical Limitations**: While the MaxEnt provides a theoretical foundation for policy alignment, its real-world implementation in RLHF suffers from critical failure modes. These failures stem directly from the divergence between theoretical assumptions and practical proxy systems.

1. The Epistemological Failure (The Proxy Reward Problem): In the original formulation, the feature constraint function $f(x)$ represents absolute, infallible empirical statistics (e.g., physical laws). MaxEnt assumes that the constraints we force the policy to satisfy are ground-truth facts. In practical RLHF, the constraint $R(q,o)$ is replaced by a learned Proxy Reward Model ($R_\phi$) trained on a finite dataset of human preferences. Because the optimization system executes the MaxEnt principle flawlessly, it optimizes for the exact geometry of $R_\phi$, including all of its adversarial blind spots and overfitting artifacts. This is the root cause of **Reward Hacking**, where the policy does not break the laws of Information Theory; it obeys a corrupted constraint too well.

2. The Exponential Exploitation via Policy Drift: The theoretical optimal solution dictates an exponential scaling of the reward signal: $\exp\left(\frac{R(q,o)}{\beta}\right)$. This creates a severe vulnerability due to the dynamic nature of RL training:

* **The Setup:** The Reward Model $R_\phi$ is trained on data generated by the initial SFT model ($\pi_{\text{ref}}$) and remains frozen.
* **The Drift:** As the policy $\pi_\theta$ optimizes during the RL phase, it shifts away from $\pi_{\text{ref}}$ and enters new token spaces to maximize scores. 
* **The Exploitation:** These new token patterns become Out-of-Distribution (OOD) for the frozen Reward Model. If $R_\phi$ erroneously extrapolates an absurdly high score for an OOD pattern, the $\exp(\cdot)$ term amplifies this error exponentially, forcing the policy to abandon the prior and cheat the system.

3. Stylistic Alignment vs. Substantive Correctness: The standard MaxEnt expected value constraint ($\mathbb{E}_{\pi}[R(q,o)] = c$) forces the distribution to tilt toward features preferred by the RM. However, learned Reward Models are notoriously biased toward stylistic features (e.g., length, sycophancy, polite formatting) rather than substantive correctness (e.g., mathematical truth, logical consistency). Consequently, the framework structurally deforms the LLM into a fluent but potentially **hallucinating model** (maximizing the superficial constraint criteria while degrading the underlying reasoning capabilities).

### RL with Verifiable Reward (RLVR)



## References

- <a id="ref-schulman2017a"></a>
John Schulman, Xi Chen, and Pieter Abbeel. *Equivalence Between Policy Gradients and Soft Q-Learning*. arXiv preprint arXiv:1704.06440. https://arxiv.org/abs/1704.06440, 2017a.