# PPO for LLM Post-Training

This note summarizes the key flow of **Proximal Policy Optimization (PPO)** and how it is implemented for **language models** in the [HuggingFace TRL](https://github.com/huggingface/trl) library.

## PPO Overview
PPO is a policy gradient optimization method in RL which maximizes expected rewards while avoiding unstable update with a clipping mechanism. Given an **environment** $$(\mathcal{S}, \mathcal{A}, P, r, \gamma)$$ where:
- $\mathcal{S}$: state space  
- $\mathcal{A}$: action space  
- $P(s'|s,a)$: transition probability  
- $r(s,a)$: reward function  
- $\gamma \in [0,1]$: discount factor

The sampling process start at each timestep $t$:

**Agent observes** the current state: $$s_t \in \mathcal{S}$$

**Policy network outputs** an action sampled from its distribution: $$a_t \sim \pi_\theta(a_t | s_t)$$

**Environment transitions** to the next state and gives a reward $$s_{t+1} \sim P(\cdot | s_t, a_t), \quad r_t = r(s_t, a_t)$$

**Store trajectory** $(s_t, a_t, r_t, s_{t+1})$ in a buffer.
---

### 🎯 Policy Optimization

Goal: maximize expected discounted return  
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \gamma^t r_t\right]
$$

PPO updates the policy by comparing **new** and **old** log-probabilities:

$$
r_t(\theta) = 
\frac{\pi_\theta(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)}
= \exp(\log \pi_\theta(a_t | s_t) - \log \pi_{\text{old}}(a_t | s_t))
$$

Then optimizes the **clipped objective**:

$$
L^{\text{PPO}} = 
\mathbb{E}_t
\left[
\min\big(
r_t A_t,\;
\text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t
\big)
\right]
$$

---

### ⚙️ Advantage Estimation (GAE)

To stabilize training, advantages are computed with **Generalized Advantage Estimation (GAE)**:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
$$
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

---
