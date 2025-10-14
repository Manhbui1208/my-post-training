# PPO for LLM Post-Training

This note summarizes the key flow of **Proximal Policy Optimization (PPO)** and how it is implemented for **language models** in the [HuggingFace TRL](https://github.com/huggingface/trl) library.

## PPO Overview
PPO is a policy gradient optimization method in RL which maximizes expected rewards while avoiding unstable update with a clipping mechanism. 
