# Mathematical MDP Formulation - Inventory Management System

## 📐 Markov Decision Process Definition

A **Markov Decision Process (MDP)** is formally defined as a tuple:

**$MDP = (S, A, P, R, \gamma)$**

Where:

- **$S$**: State space
- **$A$**: Action space
- **$P$**: Transition probability function
- **$R$**: Reward function
- **$\gamma$**: Discount factor

Let's define each component for our inventory management problem.

---

## 🎯 1. State Space ($S$)

The underlying inventory system is a **Partially Observable Markov Decision Process (POMDP)** due to unobservable lead times.

To enable the use of standard MDP-based reinforcement learning algorithms, we construct an augmented state representation by stacking the last $k+1$ observations.

This augmented state serves as an approximate belief state, allowing the agent to infer latent dynamics and yielding a process that is **approximately Markov**.

### Mathematical Definition

At each decision epoch $t$ (beginning of day $t$), the environment emits an observation vector $o_t \in \mathbb{R}^4$:

**$$o_{t, j} = \bigl(I_{t,j}, O_{t,j}\bigr)$$**

Where for each product $j \in \{0, 1\}$:
- **$I_{t,j} \in \mathbb{Z}$**: Net Inventory levels (`on-hand - backorders`) where:
  - $I_{t,j} > 0$: On-hand inventory ($I^+$)
  - $I_{t,j} < 0$: Backlogged demand ($I^-$)
- **$O_{t,j} \in \mathbb{N}$**: Outstanding (in-transit) orders

The **state** at time $t$ is defined as a sequence of the most recent $k+1$ observations:

**$s_t = \bigl[o_t, o_{t-1}, \dots, o_{t-k}\bigr]$**

Thus, the state is a $(k+1) \times 4$-dimensional vector, flattened when used as input to the neural network.

### Inventory Position

For completeness, the inventory position for product $j$ at time $t$ is defined as:
**$IP_j(s) = I_j - B_j + O_j$**

This represents the effective inventory level accounting for backorders and incoming orders. While not explicitly included as a separate state variable, this quantity is implicitly available to the agent through the stacked observations.

### Continuous vs Discrete

- **Conceptually: Discrete** ($\mathbb{Z}^{16}$) representing physical units of inventory and orders (4 features $\times$ 4 stacked frames).
- **Implementation: Continuous** ($\mathbb{R}^{16}$) as required for the neural network input layer.
- **Normalization**:Instead of fixed bounds, we apply Running Mean-Variance Normalization using the `VecNormalize` wrapper from Stable-Baselines32. The normalized state $\tilde{s}_t$ is computed as:

  $$\tilde{s}_t = \frac{s_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}}$$

  Where $\mu_t$ (running mean) and $\sigma_t^2$ (running variance) are updated automatically at every step based on the history of observed states. This ensures the neural network always receives inputs centered around 0 with unit variance, regardless of how large the inventory grows.

---

## 🎬 2. Action Space (A)
The action space represents the replenishment decisions made by the agent at the beginning of each day. Since the system manages two distinct products simultaneously, the action is a multi-dimensional vector representing independent order quantities for each product.

### Mathematical Definition

At each decision epoch $t$, the agent selects an action vector $a_t \in \mathbb{N}^2$:

**$a_t = (q_{t,0}, q_{t,1})$**

Where for each product $j \in \{0, 1\}$: 
- $q_{t,j} \in \{0, 1, \dots, Q_{max}\}$: The quantity of units to order from the supplier for product $j$ at time $t$.

## Action Bounds and Constraints

The action space is discrete and bounded. We define a maximum order quantity $Q_{max}$ to limit the search space to a feasible range:

**$$A = \{ a \in \mathbb{Z}^2 \mid 0 \le q_{t,j} \le Q_{max} \quad \forall j \in \{0, 1\} \}$$**


- $q_{t,j} = 0$: No order is placed for product $j$ (Corresponds to "Review but do not order") at time $t$.
- $q_{t,j} > 0$: An order of $q_{t,j}$ units is placed immediately.

### Hyperparameter: Maximum Order Quantity ($Q$)

The parameter $Q$ defines the upper bound of the action space. This is a critical hyperparameter that requires tuning:
- Too Small ($Q < \text{Optimal Batch}$): Handicaps the agent by preventing it from placing sufficiently large orders to cover demand spikes or long lead times.
- Too Large ($Q \gg \text{Demand}$): Unnecessarily increases the size of the action space (exploration difficulty), potentially slowing down the learning process as the agent wastes time exploring uselessly large order quantities.
---

## 🔄 3. Transition Dynamics (P)

The transition dynamics $P(s_{t+1} | s_t, a_t)$ describe how the system evolves from the current state $s_t$ to the next state $s_{t+1}$ given an action $a_t$. In this reinforcement learning context, the transition function is model-free (unknown to the agent) and is implicitly defined by the Discrete Event Simulation (DES) logic implemented in SimPy.

### Transition Function

The transition probability is:

**$P(s_{t+1} | s_t, a_t) = P(s' | s, a)$**

This represents the probability of reaching state $s'$ from state $s$ after taking action $a$.

### Stochastic Elements

The environment is stochastic and is driven by two primary sources of randomness defined in the assignment:

1. **Demand ($D_{t,j}$)**

   **Arrivals**: Customer orders arrive according to an exponential distribution with rate $\lambda = 0.1$.

    **Size**: Each arrival requests a specific quantity $d$ based on the product type:

    - **Product 1**: D ∈ {1, 2, 3, 4} with probabilities $\{\frac{1}{6}, \frac{1}{3}, \frac{1}{3}, \frac{1}{6}\}$
    - **Product 2**: D ∈ {5, 4, 3, 2} with probabilities $\{\frac{1}{8}, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}\}$

2. **Lead Time ($L_j$)**

    The time delay between placing an order and receiving it is modeled as a continuous random variable:

    1. **Product 1**: $L \sim U(0.5, 1.0)$ month
    2. **Product 2**: $L \sim U(0.2, 0.7)$ month.

    >**Note:** The lead time is handled internally by the simulator and is not directly observable by the agent.

### State Update

The transition from state $s_t$ to $s_{t+1}$ involves two distinct updates: the physical inventory update (simulation physics) and the agent's observation stack update.

1. **Physical Inventory Update**
At the beginning of day $t$, after the agent selects action $a_t = [q_{t,1}, q_{t,2}]$ (order quantities), the simulation proceeds as follows:
    - **Order Placement** If $q_{t,j} > 0$, an order is triggered. The "On-Order" quantity increases immediately:
      $$O_{t,j} \leftarrow O_{t,j} + q_ {t,j}$$
      The simulator schedules a delivery event to occur at time $t + L_j$, where $L_j$ is sampled from the specific uniform distribution.
    - **Order Arrivals (Deliveries)**: Let $Q_{arr, j}$ be the total quantity of products from previous orders that arrive during the interval $[t, t+1)$. These units are added to the net inventory and removed from the outstanding count:
    
      $$I'_{t,j} \leftarrow I_{t,j} + Q_{arr, j}$$
      $$O_{t+1,j} \leftarrow O_{t,j} - Q_{arr, j}$$
      
      > Note: As per assignment rules, arriving units are first used to satisfy any existing backlog.
    - **Demand Fulfillment** The accumulated demand $D_{t,j}$ is subtracted from the net inventory:
      $$I_{t+1,j} = I'_{t,j} - D_{t,j}$$

2. **Observation Construction**The agent observes the new system state. Consistent with our state definition, we use the Net Inventory directly:
   $$o_{t+1} = [I_{t+1,0}, O_{t+1,0}, I_{t+1,1}, O_{t+1,1}]$$
    For each product $j$:
    - $I_{t+1,j} \in \mathbb{Z}$: Captures both on-hand (positive) and backlog (negative) levels.
    - $O_{t+1,j} \in \mathbb{N}$: The updated count of orders still in transit.



3. **State Stack Update ($s_{t+1}$)** Finally, the full state $s_{t+1}$ is constructed by shifting the frame stack to include the newest observation and discard the oldest:$$s_{t+1} = [o_{t+1}, o_t, \dots, o_{t-k+1}]$$



---

## 💰 4. Reward Function (R)

### Mathematical Definition

The immediate reward $r_t$ received after taking action $a_t$ and transitioning to state $s_{t+1}$ is defined as:

$$r_t = - C_{total}(t) = - \sum_{j=0}^{1} \left( C_{order}^{(j)}(t) + C_{holding}^{(j)}(t) + C_{shortage}^{(j)}(t) \right)$$

We use **negative cost** as reward (minimizing cost = maximizing reward).

### Cost Components

The total cost is composed of three distinct penalties defined by the assignment specifications:

1. **Ordering Costs ($C_{order}$)** 

    Incurred whenever a replenishment order is placed with the supplier. It consists of a fixed setup cost ($K$) and a variable incremental cost ($i$) per unit ordered.
    
    $$C_{order}^{(j)}(t) = \begin{cases} K + i \cdot q_{t,j} & \text{if } q_{t,j} > 0 \\ 0 & \text{if } q_{t,j} = 0 \end{cases}$$

    - **Setup Cost ($K$)** Fixed cost for placing an order (e.g., administrative or transport fees).
    - **Incremental Cost ($i$)**: Cost per unit of item purchasing.
    
2. **Holding Costs ($C_{holding}$)** 
    
    Incurred for storing inventory in the warehouse. It applies only to positive on-hand inventory levels.
    $$C_{holding}^{(j)}(t) = h \cdot I_{t+1, j}^{+} = h \cdot \max(0, I_{t+1, j})$$
    - **Holding Cost ($h$)**: Cost per unit per time period (e.g., storage space, insurance).
  
3. **Shortage Costs ($C_{shortage}$)** 

    Incurred when demand cannot be satisfied immediately, leading to a backlog. It applies only to negative inventory levels.

    $$C_{shortage}^{(j)}(t) = \pi \cdot I_{t+1, j}^{-} = \pi \cdot \max(0, -I_{t+1, j})$$

    - **Penalty Cost ($\pi$)**: Cost per unit backlogged (e.g., loss of goodwill, expedited shipping).

### Parameter Configuration
The cost parameters specified for the assignment are as follows:


| Parameter         | Symbol | Value | Description                                  |
|-------------------|--------|-------|----------------------------------------------|
| Setup Cost        | K      | 10.0  | Fixed cost per order placed                  |
| Incremental Cost  | i      | 3.0   | Variable cost per unit                       |
| Holding Cost      | h      | 1.0   | Cost per unit held per day                   |
| Shortage Cost     | $\pi$     | 7.0   | Penalty per unit backlogged per day           |

### ⚙️ Reward Normalization (Implementation Detail)

The raw reward function $r_t = -C_{total}(t)$ yields values with high variance and large magnitudes (e.g., varying between $-10$ and $-500$). To ensure numerical stability and efficient gradient descent during neural network training, we apply Running Return Normalization.Instead of using the raw $r_t$ directly, the agent receives a normalized reward 
$\hat{r}_t$:$$\hat{r}_t = \frac{r_t}{\sqrt{\sigma_{ret}^2 + \epsilon}}$$

- **Implementation:** We utilize the `VecNormalize` wrapper from the Stable-Baselines3 library.

## 🎲 5. Discount Factor ($\gamma$)

**$\gamma \in [0, 1]$**
- $γ = 0$: Only immediate reward matters (myopic/shortsighted)
- $γ → 1$: Future rewards matter more (farsighted)

 ### Effective Horizon
 
 The discount factor implicitly defines the "Effective Horizon" ($H_{eff}$) of the agent, which approximates how many future steps significantly influence the current decision:
 $$H_{eff} \approx \frac{1}{1 - \gamma}$$
 - $\gamma = 0.95$: Horizon $\approx 20$ days.
 - $\gamma = 0.99$: Horizon $\approx 100$ days.
 - $\gamma = 0.999$: Horizon $\approx 1000$ days.

### Application to Inventory Control

The agent must be farsighted.

Inventory control is inherently a long-term planning problem with delayed consequences:
- **Lead Time Delay**: An order placed today incurs an immediate ordering cost ($K + i \cdot q$) but does not replenish inventory until $t + L$. A myopic agent would see the cost but not the benefit.
- **Long-term Ripple Effects**: Ordering too little today $\to$ Stockouts (Shortage Costs) next week.Ordering too much today $\to$ Excess stock (Holding Costs) for potentially months.

Therefore, we recommend setting $\gamma$ in the high range:
- **Recommended Range**: $\gamma \in [0.99, 0.999]$ 
- **Reasoning**: This ensures the agent accounts for the full cycle of ordering, waiting, and selling, preventing it from greedily avoiding setup costs at the expense of massive future shortages.


## 🎯 Optimization Objective
The goal of the reinforcement learning agent is to find an optimal policy $\pi^*$ that minimizes the long-term operational cost. In the standard RL maximization framework, this is equivalent to maximizing the expected cumulative discounted reward (negative cost).

### Value Functions

The **state value function** under policy $\pi$ is:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s \right]$$

The expected return taking action $a$ in state $s$, and thereafter following policy $\pi$.

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a \right]$$

### Optimal Policy

Find policy $\pi^*$ that maximizes expected cumulative reward:

**$\pi^* = \arg\max_\pi \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)]$**

### Bellman Optimality Equation

**$V^*(s) = \max_a [R(s, a) + \gamma \cdot \sum_{s'} P(s'|s,a) V^*(s')]$**
**$Q^*(s, a) = R(s, a) + \gamma \cdot \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')$**

---

## 🔢 Problem Characteristics

### 1. State Space

  - **Nature: Partially Observable (POMDP)**. The raw observation $o_t$ does not fully capture the system state due to hidden lead times.
  - **Resolution Strategy: Frame Stacking.** By defining the state $S_t$ as a sequence of the last 4 observations (current + k history), we construct a "belief state" that allows the agent to infer hidden temporal dynamics.
  - **Dimensionality:**
    - **Raw Observation ($o_t$):** 4 features ($2 \times$ Inventory_level, $2 \times$ On-Order).
    - **Effective State Input ($S_t$):** 16 features. Calculated as: $(k+1) \times |o_t| = 4 \times 4 = 16$

### 2. Action Space

- **Type**: Multi-Discrete
- **Dimensionality**: 2
- **Size**: With $Q_{max}=20$, the total number of unique action combinations is 441 ($21 \times 21$).

### 3. Transition Dynamics

- **Type**: Stochastic
- **Markov Property**: ✅ Approximately satisfied via state augmentation
- **Model-free or Model-based**: We have access to simulator (model-based possible)

### 4. Reward

- **Type**: Dense (received every step)
- **Structure**: Negative cost
- **Bounded**: No (costs can grow arbitrarily)

### 5. Horizon

- **Type**: Infinite (continuing task)
- **Episodes**: We simulate finite episodes for training

---

## 📊 MDP Properties

### Markov Property Validity

$$P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \dots) \approx P(S_{t+1} | S_t, A_t)$$
- **Raw Observation ($o_t$)**: Does NOT hold. Knowing only the current inventory and order count is insufficient to predict arrival times.
- **Stacked State ($S_t$):** Effectively Holds. By including the history stack in $S_t$, the probability of the next state (including likely arrivals) depends almost entirely on the information contained within the current stack, approximately recovering the Markov property required for RL algorithms.

### Stationary

**$P(s' | s, a)$** and **$R(s, a)$** do not change over time.

✅ **Holds**: The system dynamics are time-invariant.

### Episodic vs Continuing

- **Implementation**: Episodic (finite-length episodes for training)
- **Reality**: Continuing task (inventory management never "ends")

---


## 📈 Curse of Dimensionality

### State Space Explosion

With our updated observation definition ($Inv, OnOrder$ for 2 products), a single observation frame has 4 dimensions. With a frame stack of 3, the effective input state vector  lies in $\mathbb{R}^{16}$.

Even with a coarse discretization (e.g., 50 bins per variable), the size of the state space for tabular methods would be:
$$|S| \approx 50^{16} \approx 1.5 \times 10^{27} \text{ states}$$

This astronomical number renders tabular methods (like Q-Learning or SARSA) strictly impossible. This necessitates **Deep Reinforcement Learning (Function Approximation)**, where a neural network learns to generalize value estimates across this continuous high-dimensional space.

### Action Space Implications

The choice of algorithm significantly impacts how the action space is handled:

* **DQN (Flat Space):**
Standard DQN requires a flattened discrete action space. With $Q_{max}=20$ , we have 21 choices per product.
$$|A| = 21 \times 21 = 441 \text{ discrete actions}$$

The neural network must output 441 distinct Q-values. While feasible, exploring 441 actions randomly can be slow, and the "max" operator over a large vector can introduce maximization bias.
* **PPO (Multi-Discrete):**
PPO supports factorized action spaces (`MultiDiscrete`). The network outputs two independent probability distributions:
$$Output_{size} = 21 + 21 = 42 \text{ logits}$$

This factorization ($42 \ll 441$) makes the policy significantly more sample-efficient to learn, as the agent understands that changing the order quantity for Product A doesn't necessarily invalidate its knowledge about Product B.