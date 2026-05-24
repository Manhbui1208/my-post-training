# Note on GRPO Done Right (DR. GRPO)

- Most base models already exhibit the self-reflection abilities (e.g., Aha moments) during pre-training (The hypothesis is "those base models are pre-trained with concatenated question-answers")
- RL helps to amplify those self-reflection behaviors and the correct responses also increase after RL-tuned
- The response length also substantially increase in both correct/incorrect answers 
- The average length of incorrect responses are notably longer than correct ones (perhaps due to hard questions require longer reasoning chains?)
- Self-reflection does not correlate to the increased accuracy during inference (Though the self-reflection can improve exploration during training) => this behavior is not clean and accurate to explain the improved accuracy after RL.

## Basics of RL for Language Models
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

## 1. State

At generation step `t`, the state is the concatenation of:

1. the input question/prompt `q`
2. the output tokens generated so far `o_<t`

```math
s_t = q ; o_{<t}
```

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

## 2. Action

At each step, the policy model selects the next token from the vocabulary: $o_t \in \mathcal{A}$.

The policy is:

```math
\pi_\theta(\cdot | s_t)
```

This means:

> Given the current context `s_t`, the LLM outputs a probability distribution over possible next tokens.

Then one token is selected, usually by sampling during RL training.

---

## 3. Deterministic transition

After the next token `o_t` is selected, the next state is obtained by appending that token to the current context:

```math
s_{t+1} = s_t ; [o_t]
```

This transition is **deterministic** because once the token is chosen, the next state is fixed.

Important distinction:

- The **policy** can be stochastic because it samples the next token.
- The **transition** is deterministic because the chosen token is simply appended to the context.

---

## 4. Initial state

The generation process starts by sampling a question from the question distribution:

```math
s_1 = q \sim p_Q
```

In simple terms:

> The prompt/question is the initial state of the episode.

---

## 5. End of trajectory

The trajectory ends when either: (1) the model generates the `[eos]` token, or (2) the model exhausts the generation budget.

A full trajectory is the full generated response:

```math
q ; o
```

where:

```math
o = [o_1, o_2, ..., o_T]
```

---