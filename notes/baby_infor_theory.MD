## Baby Language of Information Theory: From Entropy to KL Divergence

Information theory gives us a way to think about probability distributions in terms of **uncertainty**, **coding cost**, and **information difference**.

---

### 1. Entropy: Uncertainty inside one distribution

Entropy measures how uncertain or spread out a probability distribution is.

$$
H(p) = -\sum_x p(x)\log p(x)
$$

Intuitively:

> Entropy tells us how many bits, on average, are needed to describe an outcome sampled from distribution $p$.

If the distribution is very concentrated, entropy is low because the outcome is easy to predict.  
If the distribution is spread out, entropy is high because many outcomes are still possible.

Examples:

- If a coin always gives Head, entropy is low.
- If a coin is fair, entropy is higher.
- If a language model spreads probability across many possible next tokens, entropy is high.
- If a language model is almost certain about one next token, entropy is low.

Baby language:

> Entropy asks: “How surprised should I expect to be before seeing the outcome?”

So entropy measures **uncertainty inside one distribution**.

---

### 2. Cross-Entropy: Coding data with another distribution

Suppose the true data distribution is $p$, but we use another distribution $q$ to represent or encode it.

Cross-entropy measures the average coding cost when the data actually comes from $p$, but we encode it using $q$.

$$
H(p,q) = -\sum_x p(x)\log q(x)
$$

Intuitively:

> Cross-entropy tells us how expensive it is (average bits) to encode data from $p$ using a model distribution $q$.

If $q$ matches $p$, the coding cost is low.  
If $q$ assigns low probability to events that often happen under $p$, the coding cost becomes high.

Baby language:

> Cross-entropy asks: “If my model believes $q$, how many bits average do I need to describe reality $p$?”

---

### 3. KL Divergence: Information difference from a prior/model

KL divergence measures how different one distribution $p$ is from another distribution $q$.

$$
D_{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}
$$

It can also be connected to entropy and cross-entropy:

$$
H(p,q) = H(p) + D_{KL}(p\|q)
$$

Therefore:

$$
D_{KL}(p\|q) = H(p,q) - H(p)
$$

Intuitively:

> KL divergence is the extra number of bits needed when we use $q$ to encode data that actually comes from $p$, compared to using the correct distribution $p$.

So:

- Entropy measures uncertainty **inside one distribution**.
- KL divergence measures information difference **between two distributions**.

Baby language:

> Entropy asks: “How uncertain is this distribution?”  
> KL asks: “How much extra information is needed because my model/prior distribution is different from the true/new distribution?”

---

### 4. Entropy vs KL: The key distinction

Entropy:

$$
H(p)
$$

measures how uncertain distribution \(p\) is.

KL divergence:

$$
D_{KL}(p\|q)
$$

measures how much distribution $p$ differs from distribution $q$.

In simple terms:

> Entropy is about uncertainty.  
> KL is about information shift from one distribution to another.

If we use log base 2, both entropy and KL can be measured in **bits**.  

---

### 5. Why this matters in ML/DL

In supervised learning, we usually have:

- $p$: the true label distribution
- $q_\theta$: the model prediction distribution

Training with cross-entropy loss means:

$$
\mathcal{L}_{CE} = -\sum_x p(x)\log q_\theta(x)
$$

Since:

$$
H(p,q_\theta) = H(p) + D_{KL}(p\|q_\theta)
$$

and $H(p)$ is fixed with respect to the model, minimizing cross-entropy is equivalent to minimizing:

$$
D_{KL}(p\|q_\theta)
$$

So in ML/DL:

> Cross-entropy training pushes the model prediction distribution $q_\theta$ closer to the true data distribution $p$.

---

### 6. Simple classification example

Suppose the true label is “cat”.

The target distribution is:

$$
p = [1, 0, 0]
$$

The model predicts:

$$
q_\theta = [0.7, 0.2, 0.1]
$$

The cross-entropy loss is:

$$
\mathcal{L}_{CE} = -\log(0.7)
$$

If the model becomes more confident in the correct class:

$$
q_\theta = [0.95, 0.03, 0.02]
$$

then the loss becomes:

$$
\mathcal{L}_{CE} = -\log(0.95)
$$

which is smaller.

Baby language:

> The model is punished when its predicted distribution does not match the true label distribution. The more probability it gives to the correct label, the lower the loss.

---

### 7. Why Maximizing Entropy is Related to Minimizing KL

At first, entropy and KL divergence look like two different concepts:

- Entropy $H(p)$ measures uncertainty inside one distribution.
- KL divergence $D_{KL}(p\|q)$ measures how different one distribution is from another.

However, maximizing entropy can be seen as a special case of minimizing KL divergence.

Assume we have a finite outcome space $\mathcal{X}$, and the uniform distribution over this space is:

$$
U(x) = \frac{1}{|\mathcal{X}|}
$$

The KL divergence from $p$ to the uniform distribution is:

$$
D_{KL}(p\|U)
=
\sum_x p(x)\log\frac{p(x)}{U(x)}
$$

Substitute $U(x)=\frac{1}{|\mathcal{X}|}$:

$$
D_{KL}(p\|U)
=
\sum_x p(x)\log\frac{p(x)}{1/|\mathcal{X}|}
$$

$$
=
\sum_x p(x)\log p(x)
+
\sum_x p(x)\log|\mathcal{X}|
$$

Since:

$$
\sum_x p(x)=1
$$

we get:

$$
D_{KL}(p\|U)
=
\sum_x p(x)\log p(x)
+
\log|\mathcal{X}|
$$

Because entropy is:

$$
H(p) = -\sum_x p(x)\log p(x)
$$

therefore:

$$
D_{KL}(p\|U)
=
\log|\mathcal{X}| - H(p)
$$

Since $\log|\mathcal{X}|$ is constant, minimizing $D_{KL}(p\|U)$ is equivalent to maximizing \(H(p)\).

$$
\arg\min_p D_{KL}(p\|U)
=
\arg\max_p H(p)
$$

Intuitively:

> Maximizing entropy means making the distribution as spread out and uncommitted as possible.  
> Minimizing KL to the uniform distribution means making the distribution as close as possible to the most neutral distribution.

So, in a finite discrete space:

> Maximum entropy is the same as minimum KL divergence to the uniform prior.

Baby language:

> If we know nothing, the safest prior is uniform: treat every outcome equally.  
> Maximizing entropy says: “do not become more confident than the evidence allows.”  
> Minimizing KL to uniform says: “stay as close as possible to equal belief over all outcomes.”  
> These are two ways of saying the same thing.

This is why entropy regularization can be interpreted as a special case of KL regularization, where the prior distribution is uniform.

More generally:

- No informative prior:

$$
\max_p H(p)
$$

means choosing the least committed distribution.

- With an informative prior \(p_0\):

$$
\min_p D_{KL}(p\|p_0)
$$

means choosing the distribution that stays closest to the prior while satisfying the known constraints.

Therefore, entropy maximization is not separate from KL minimization. It is the special case where the prior is uniform.

Therefore, from an information-theoretic view:

> Entropy measures uncertainty.  
> Cross-entropy measures prediction or coding cost under a model.  
> KL divergence measures information shift from one distribution to another.

This is why KL is useful in RLHF and LLM RL: it acts as an information-distance cost, preventing reward optimization from moving the model too far away from the prior knowledge encoded in the reference model.