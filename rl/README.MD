# A Sketchy View on RL for LLM

## Mapping RL concepts to LLM generation

| RL concept | LLM generation meaning |
|---|---|
| State | Current context: question + generated tokens so far |
| Action | Next token |
| Policy | LLM / Transformer model |
| Transition | Append selected token to context |
| Reward | Verifier score, reward model score, or final-answer correctness |
| Trajectory | Full generated answer |
| Episode ends | `[eos]` token or max generation length |

---

## Key intuition

LLM generation is a sequential decision-making process.

At every step:

```text
current context
→ policy samples next token
→ token is appended
→ new context
→ repeat
```

So in RL language:

> The LLM is the policy, the generated tokens are actions, and the full response is a trajectory.