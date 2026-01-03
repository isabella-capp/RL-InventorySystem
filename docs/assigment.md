# Assignment: Reinforcement Learning for Inventory Management
**`stable-baseline3`**, **`gymnasyium`**, **`simpy`**, **`DQN`**, **`PPO`**

## Overview

**Expected effort:**
- Successfully implement a discrete event simulation of an operation management or supply chain environment.
- Adhere to the guidelines discussed in lectures when implementing the simulation environment.
- Conduct a thorough performance study of the system using methods discussed in lectures.
- Implement the reinforcement learning agent from scratch or use a library’s black-box implementation.
- If using a library’s black-box implementation, ensure you thoroughly understand the algorithm being used.
- You may use any extension of the RL algorithms discussed in the lectures, provided you understand the type of algorithm and its peculiarities.
- Experiment with di erent RL algorithms and di erent hyper-parameter combinations.

## Inventory System

At least one reinforcement learning agent must be used to **manage the replenishment policies** of the warehouse.

At the beginning of each day, the warehouse management must decide whether to **place a replenishment order** **for each product**. If an order is placed, the **quantity to be ordered** (in
discrete units) must also be determined.

**Objective:** minimize the overall cost.

- A company sells a **two different products**, each sourced from a different supplier
- Students must **compare the performance** of the **RL-based solution** with the performance of
the **s-min, S-max policy**, for each product.
- **Demand interrarival times** are exponential random variables with lambda $0.1$
- **$K = 10$**
- $i = 3$
- $h = 1$
- $\pi = 7$
- **Demand Distribution of the first product**
    
    $D =\begin{cases}1 & \text{w.p. } \frac{1}{6} \\2 & \text{w.p. } \frac{1}{3} \\3 & \text{w.p. } \frac{1}{3} \\4 & \text{w.p. } \frac{1}{6}\end{cases}$
    
    **Lead Time: $U(0.5;1)$**
    
- **Second Product demand distristribution**
    
    $D =\begin{cases}5 & \text{w.p. } \frac{1}{8} \\4 & \text{w.p. } \frac{1}{2} \\3 & \text{w.p. } \frac{1}{4} \\2 & \text{w.p. } \frac{1}{8}\end{cases}$
    
    **Lead Time:** $U(0.2; 0.7)$
    

>⚠️ The lead time is generated internally by the simulation environment and is **not observable by the decision-making agent**, ensuring a realistic and partially observable setting.