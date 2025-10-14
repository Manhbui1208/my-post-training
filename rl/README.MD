# PPO for LLM Post-Training

This note summarizes the key flow of **Proximal Policy Optimization (PPO)** and how it is implemented for **language models** in the [HuggingFace TRL](https://github.com/huggingface/trl) library.

## PPO Overview
PPO is a policy gradient optimization method in RL which maximizes expected rewards while avoiding unstable update with a clipping mechanism. Given an **environment** $$(\mathcal{S}, \mathcal{A}, P, r, \gamma)$$ where:
- $\mathcal{S}$: state space  
- $\mathcal{A}$: action space  
- $P(s'|s,a)$: transition probability  
- $r(s,a)$: reward function  
- $\gamma \in [0,1]$: discount factor

With the policy network $\pi_\theta$, the **sampling process** start at each timestep $t$:

1. **Agent observes** the current state: $$s_t \in \mathcal{S}$$

2. **Policy network outputs** an action sampled from its distribution: $$a_t \sim \pi_\theta(a_t | s_t)$$

3. **Environment transitions** to the next state and gives a reward $$s_{t+1} \sim P(\cdot | s_t, a_t), \quad r_t = r(s_t, a_t)$$

4. **Store trajectory** $(s_t, a_t, r_t, s_{t+1})$ in a buffer.

After finishing sampling, the **policy optimization** start:
#### 🔸 Step 1. Compute the advantage
Measure how much better an action was compared to the expected value:

$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

(Positive $A_t$ → good action, Negative $A_t$ → bad action)
