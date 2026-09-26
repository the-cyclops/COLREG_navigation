# Autonomous Marine Navigation with COLREG Compliance via Signal Temporal Logic & Constrained RL

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Unity ML-Agents](https://img.shields.io/badge/Unity%20ML--Agents-1.1.0-black?logo=unity)](https://github.com/Unity-Technologies/ml-agents)
[![RTAMT](https://img.shields.io/badge/STL%20Monitoring-RTAMT-green)](https://github.com/nickovic/rtamt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end framework integrating **Signal Temporal Logic (STL)** runtime verification with **Constrained Multi-Critic PPO** and **Conflict-Averse Gradient Descent (CAGrad)** for safe Autonomous Surface Vessel (ASV) navigation in a high-fidelity 3D marine environment (Unity HDRP).

```mermaid
flowchart LR
    subgraph Sim ["Unity Simulation"]
        Env["3D Marine Environment<br/>(15x15m Arena, HDRP Water)"]
        Obs["Observations (37-dim)<br/>(Kinematics + LiDAR + Memory)"]
        Env --> Obs
    end

    subgraph Logic ["Formal Verification (RTAMT)"]
        CPA["CPA & Kinematics<br/>(t_cpa, d_min)"]
        STL["Dense STL Monitors<br/>(R1 Distance, R2 Speed, R6 Stand-on)"]
        Cost["Step Costs c_k<br/>c_k = tanh(-ρ_k) · α_cost"]
        CPA --> STL --> Cost
    end

    subgraph RL ["Constrained RL (Python)"]
        Critics["Decoupled Multi-Critic<br/>V^φ (Reward) + 3 Cost Critics V^ψ_k"]
        CAGrad["Lexicographic Switching &<br/>CAGrad Conflict Resolution (c=0.5)"]
        Actor["Policy π_θ(a|s)"]
        Critics --> CAGrad --> Actor
    end

    Obs --> CPA
    Obs --> Critics
    Cost --> Critics
    Actor -->|"a_t = [throttle, steer]"| Env
