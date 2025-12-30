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

The underlying inventory system is a Partially Observable Markov Decision Process (POMDP) due to unobservable lead times.

To enable the use of standard MDP-based reinforcement learning algorithms, we construct an augmented state representation by stacking the last $k+1$ observations.

This augmented state serves as an approximate belief state, allowing the agent to infer latent dynamics and yielding a process that is approximately Markov.

### Mathematical Definition

At each decision epoch $t$ (beginning of day $t$), the environment emits an observation vector $o_t \in \mathbb{R}^6$:

**$$o_t = \bigl(I_{t,0}, I_{t,1}, B_{t,0}, B_{t,1}, O_{t,0}, O_{t,1}\bigr)$$**

Where for each product $i \in \{0, 1\}$:
- **$I_{t,0}, I_{t,1} \in \mathbb{Z}$**: On-hand inventory levels for products 0 and 1
- **$B_{t,0}, B_{t,1} \in \mathbb{Z}_+$**: Backlog (unsatisfied demand)
- **$O_{t,0}, O_{t,1} \in \mathbb{Z}_+$**: Outstanding (in-transit) orders

The **state** at time $t$ is defined as a sequence of the most recent $k+1$ observations:

**$s_t = \bigl[o_t, o_{t-1}, \dots, o_{t-k}\bigr]$**

with $k = 3$ in our implementation.

Thus, the state is a $(k+1) \times 6$-dimensional vector, flattened when used as input to the neural network.

### State Space Bounds

**$S = \{s \mid 0 \leq I_i \leq 200, 0 \leq B_i \leq 100, 0 \leq O_i \leq 150\}$** for $i \in \{0,1\}$.  
Note: Negative inventory levels are represented implicitly through the backlog variable $B_i = \max(0, -Inv_i)$

### Inventory Position

For completeness, the inventory position for product $i$ at time $t$ is defined as:

**$IP_i(s) = I_i - B_i + O_i$**

This represents the effective inventory level accounting for backorders and incoming orders. While not explicitly included as a separate state variable, this quantity is implicitly available to the agent through the stacked observations.

### Continuous vs Discrete

- **Implementation**: Continuous (ℝ⁶) for neural networks
- **Conceptually**: Discrete (ℤ⁶) for units of inventory
- **Normalization**: For neural networks, we normalize:

  **$\tilde{s} = (s - \mu) / \sigma$**

  where μ = (50, 50, 5, 5, 20, 20) and σ = (40, 40, 15, 15, 30, 30)

---

## 🎬 2. Action Space (A)
The action space represents the replenishment decisions made by the agent at the beginning of each day. Since the system manages two distinct products simultaneously, the action is a multi-dimensional vector representing independent order quantities for each product.

### Mathematical Definition

At each decision epoch $t$, the agent selects an action vector $a_t \in \mathbb{N}^2$:

**$a_t = (a_{t,0}, a_{t,1})$**

Where for each product $i \in \{0, 1\}$: 
- $a_{t,i} \in \{0, 1, \dots, Z_{max}\}$: The quantity of units to order from the supplier for product $i$.

## Action Bounds and Constraints

The action space is discrete and bounded. We define a maximum order quantity $Z_{max}$ to limit the search space to a feasible range, given that demand per period is relatively low (max 4 or 5 units per day).

**$$A = \{ a \in \mathbb{Z}^2 \mid 0 \le a_{t,i} \le Z_{max} \quad \forall i \in \{0, 1\} \}$$**

In our implementation, we set $Z_{max} = 20$.
- $a_{t,i} = 0$: No order is placed for product $i$ (Corresponds to "Review but do not order").
- $a_{t,i} > 0$: An order of $a_{t,i}$ units is placed immediately.

### Hyperparameter: Maximum Order Quantity ($Z$)

The parameter $Z$ defines the upper bound of the action space. This is a critical hyperparameter that requires tuning:
- Too Small ($Z < \text{Optimal Batch}$): Handicaps the agent by preventing it from placing sufficiently large orders to cover demand spikes or long lead times.
- Too Large ($Z \gg \text{Demand}$): Unnecessarily increases the size of the action space (exploration difficulty), potentially slowing down the learning process as the agent wastes time exploring uselessly large order quantities.

---

## 🔄 3. Transition Dynamics (P)

The transition dynamics $P(s_{t+1} | s_t, a_t)$ describe how the system evolves from the current state $s_t$ to the next state $s_{t+1}$ given an action $a_t$. In this reinforcement learning context, the transition function is model-free (unknown to the agent) and is implicitly defined by the Discrete Event Simulation (DES) logic implemented in SimPy.

### Transition Function

The transition probability is:

**$P(s_{t+1} | s_t, a_t) = P(s' | s, a)$**

This represents the probability of reaching state $s'$ from state $s$ after taking action $a$.

### Stochastic Elements

The environment is stochastic and is driven by two primary sources of randomness defined in the assignment:

1. **Demand ($D_{t,i}$)**

   **Arrivals**: Customer orders arrive according to an exponential distribution with rate $\lambda = 0.1$.

    **Size**: Each arrival requests a specific quantity $d$ based on the product type:

    - **Product 1**: D ∈ {1, 2, 3, 4} with probabilities $\{\frac{1}{6}, \frac{1}{3}, \frac{1}{3}, \frac{1}{6}\}$
    - **Product 2**: D ∈ {5, 4, 3, 2} with probabilities $\{\frac{1}{8}, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}\}$

2. **Lead Time ($L_i$)**

    The time delay between placing an order and receiving it is modeled as a continuous random variable:

    1. **Product 1**: $L \sim U(0.5, 1.0)$ days
    2. **Product 2**: $L \sim U(0.2, 0.7)$ days.

    **Note:** The lead time is handled internally by the simulator and is not directly observable by the agent.

### State Update

The transition from state $s_t$ to $s_{t+1}$ involves two distinct updates: the physical inventory update (simulation physics) and the agent's observation stack update.

1. **Physical Inventory Update**
At the beginning of day $t$, after the agent selects action $a_t = [a_{t,1}, a_{t,2}]$ (order quantities), the simulation proceeds as follows:
    - **Order Placement** If $a_{t,i} > 0$, an order is triggered. The "On-Order" quantity increases immediately:
      $$O_{t,i} \leftarrow O_{t,i} + a_ {t,i}$$
      The simulator schedules a delivery event to occur at time $t + L_i$, where $L_i$ is sampled from the specific uniform distribution.
    - **Order Arrivals (Deliveries)**: For any previously placed orders scheduled to arrive during $[t, t+1)$, the quantity $Q_{arr, i}$ represents the arriving quantity for product $i$. This quantity is added to the net inventory and removed from outstanding orders:
    $$Inv'_{t,i} \leftarrow Inv_{t,i} + Q_{arr, i}$$
    $$O'_{t,i} \leftarrow O_{t,i} - Q_{arr, i}$$
    
    - **Demand Fulfillment** The accumulated demand $D_{t,i}$ is subtracted from the net inventory:
    $$Inv_{t+1,i} = Inv_{t,i} - D_{t,i}$$

2. **Observation Construction** The agent does not see the raw $Inv_i$. Instead, the observation vector $o_{t+1}$ splits the Net Inventory ($Inv$) into On-Hand ($I$) and Backlog ($B$) components:
For each product $i$:
    - **On-Hand**: $I_{t+1,i} = \max(0, Inv_{t+1,i})$
    - **Backlog**: $B_{t+1,i} = \max(0, -Inv_{t+1,i})$
    - **Outstanding**: $O_{t+1,i}$ (Updated sum of pending orders)

Resulting in the observation vector: $o_{t+1, i} = [I_{t+1,i}, B_{t+1,i}, O_{t+1,i}]$.

3. **State Stack Update ($s_{t+1}$)** Finally, the full state $s_{t+1}$ is constructed by shifting the frame stack to include the newest observation and discard the oldest:$$s_{t+1} = [o_{t+1}, o_t, \dots, o_{t-k+1}]$$



---

## 💰 4. Reward Function (R)

### Mathematical Definition

The immediate reward $r_t$ received after taking action $a_t$ and transitioning to state $s_{t+1}$ is defined as:

$$r_t = - C_{total}(t) = - \sum_{i=0}^{1} \left( C_{order}^{(i)}(t) + C_{holding}^{(i)}(t) + C_{shortage}^{(i)}(t) \right)$$

We use **negative cost** as reward (minimizing cost = maximizing reward).

### Cost Components

The total cost is composed of three distinct penalties defined by the assignment specifications:

1. **Ordering Costs ($C_{order}$)** 

    Incurred whenever a replenishment order is placed with the supplier. It consists of a fixed setup cost ($K$) and a variable incremental cost ($in$) per unit ordered.
    
    $$C_{order}^{(i)}(t) = \begin{cases} K + in \cdot a_{t,i} & \text{if } a_{t,i} > 0 \\ 0 & \text{if } a_{t,i} = 0 \end{cases}$$

    - **Setup Cost ($K$)** Fixed cost for placing an order (e.g., administrative or transport fees).
    - **Incremental Cost ($in$)**: Cost per unit of item purchasing.
    
2. **Holding Costs ($C_{holding}$)** 
    
    Incurred for storing inventory in the warehouse. It applies only to positive on-hand inventory levels.
    $$C_{holding}^{(i)}(t) = h \cdot I_{t+1, i}^{+} = h \cdot \max(0, I_{t+1, i})$$
    - **Holding Cost ($h$)**: Cost per unit per time period (e.g., storage space, insurance).
  
3. **Shortage Costs ($C_{shortage}$)** 

    Incurred when demand cannot be satisfied immediately, leading to a backlog. It applies only to negative inventory levels.

    $$C_{shortage}^{(i)}(t) = \pi \cdot I_{t+1, i}^{-} = \pi \cdot \max(0, -I_{t+1, i})$$
    - **Penalty Cost ($\pi$)**: Cost per unit backlogged (e.g., loss of goodwill, expedited shipping).

### Parameter Configuration
The cost parameters specified for the assignment are as follows:


| Parameter         | Symbol | Value | Description                                  |
|-------------------|--------|-------|----------------------------------------------|
| Setup Cost        | K      | 10.0  | Fixed cost per order placed                  |
| Incremental Cost  | in      | 3.0   | Variable cost per unit                       |
| Holding Cost      | h      | 1.0   | Cost per unit held per day                   |
| Shortage Cost     | pi     | 7.0   | Penalty per unit backlogged per day           |


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
  - **Resolution Strategy: Frame Stacking.** By defining the state $S_t$ as a sequence of the last 4 observations (current + 3 history), we construct a "belief state" that allows the agent to infer hidden temporal dynamics.
  - **Dimensionality:**
    - **Raw Observation ($o_t$):** 6 features ($2 \times$ On-Hand, $2 \times$ Backlog, $2 \times$ On-Order).
    - **Effective State Input ($S_t$):** 24 features. Calculated as: $(k+1) \times |o_t| = 4 \times 6 = 24$

### 2. Action Space

- **Type**: Multi-Discrete
- **Dimensionality**: 2
- **Size**: With $Z_{max}=20$, the total number of unique action combinations is 441 ($21 \times 21$).

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

**State Space Size**

Even with conservative discretization (e.g., 50 bins per variable), the raw state space size is astronomical:
$$|S_{raw}| \approx 50^6 \approx 15.6 \text{ billion states}$$

When considering the stacked state (24 dimensions), tabular methods are strictly impossible. This necessitates Deep Reinforcement Learning (Function Approximation), where a neural network learns to generalize values across similar states.

**Action Space Implications**

With 441 discrete action pairs:
- **DQN**: Would require an output layer of 441 neurons (if flattened), which can be slow to converge.
- **PPO (MultiDiscrete)**: Requires two output layers of 21 neurons each. This factorization ($2 \times 21 \ll 441$) makes PPO significantly more sample-efficient for this specific action structure.