# PPO for LLM Post-Training

This note summarizes the key flow of **Proximal Policy Optimization (PPO)** and how it is implemented for **language models** in the [HuggingFace TRL](https://github.com/huggingface/trl) library.

## PPO Overview
PPO is a policy gradient optimization method in RL which maximizes expected rewards while avoiding unstable update with a clipping mechanism. 
- Agent observes **state $s_t$**.
- Policy **πθ** outputs **action `a_t` ~ πθ(a|s_t)** (e.g., move left / generate next token).
- Environment returns:
  - **next state `s_{t+1}`**
  - **reward `r_t`**
  - **done flag** (if episode ends)

→ These tuples `(s_t, a_t, r_t, s_{t+1})` are stored in a **rollout buffer**.
