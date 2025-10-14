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
Evaluate the action in current state ($s_t$) whether it can help the actor get the good result (better reward) in the future or not.

$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Value function $V(s_t)$ aim to estimate the **expected average value** in the future (next time step) if the agent follow the strategy $(s_t, a_t)$. The advantage function (GAE) uses current reward feedback ($r_t$) additioned with the near predicted future ($V(s_{t+1})$: the expected value of the next state) and the current $V(s_t)$ to show if the current action-state can help the agent gain better results than expected or not. => (Positive $A_t$ → good action which is better than expectation from $V(s_t)$, Negative $A_t$ → bad action)

Imagination: Player (Actor) make a move (obtained $r_t$ with ($s_t, a_t$)) and the referee (Critic/Value func) said that "if you play like this, your average score will be 0.5", but the actual result (current environment feedback $r_t$ + expected value of next step $V(s_{t+1})$) is 0.8 -> actor made a good move -> increase the probability for choosing that move.

#### 🔸 Step 2. Compute the policy ratio
Compare the new and old policy probabilities:

$$
r_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)}
$$

If $r_t > 1$, the new policy thinks this action is **more likely** than before.

#### 🔸 Step 3. Apply clipping
Prevent the update from changing too much:

$$
L^{\text{PPO}} = \mathbb{E}_t \big[
\min(r_t A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t)
\big]
$$

→ Clipping keeps the policy within a “trust region” (small, stable steps).

#### 🔸 Step 4. Add value loss and entropy bonus
- **Value loss**: make $V(s_t)$ fit the true returns  
- **Entropy bonus**: encourage exploration (avoid collapse)

$$
L_\text{total} = L^{\text{PPO}} + c_v (V - R)^2 - c_e \, \mathcal{H}[\pi_\theta]
$$

#### 🔸 Step 5. Update parameters
Backpropagate $L_\text{total}$ and update both networks:
