# Aether: Generalized Traffic Engineering with Elastic Multi-agent Graph Transformers

Aether is a multi-agent traffic engineering (TE) framework with subgraph GNN feature extraction module built for learning-based approaches to wide-area network (WAN) control.

## Getting Started


### Requirements
- **torch==2.0.0**
- **gym==0.12.4** 

## Code Structure
```
Aether
├── algorithms                       # Core components for policy and RL
│   ├── EMAGT.py   # Multi-Agent Transformer model with subgraph-GNN feature extraction
│   ├── shared_buffer.py             # Replay buffer for reinforcement learning
│   ├── transformer_act.py           # Action generation module
│   ├── transformer_policy.py        # Policy definition and training
│   ├── util.py                      # General utility functions
│   └── util_algo.py                 # Algorithm-specific utilities
│   └── valuenorm.py                 # Value normalization for stabilization
│
├── env                              # Environment and topology-related tools
│   ├── experiment_utils.py          # Utilities for running experiments
│   ├── generatepath_utils.py        # Helper for generating subgraph features
│   ├── subgraph_generate.py         # Subgraph feature generation script
│   ├── te_env.py                    # Single-topology RL environment
│   └── te_env_multitopo.py          # Multi-topology RL environment
│
├── gnnencoder                       # GNN-based feature encoding modules
│   └── HirachicalGNN.py              
│
├── log                              # Logs generated during execution
│
├── config.py                        # Argument parser for input parameters
├── main.py                      # Main entry point for MAT experiments
├── trainer.py                   # Multi-Agent Transformer trainer
└── README.md                        # Project documentation
```

## Running and Evaluating Aether
Below is an example command line to train the Aether's model:

```bash
python main.py \
  --train_topos topo1 topo2 (single or multiple) \
  --test_topos topo2 (single or multiple) \
  --obj min_max_link_util (or total_flow) \
```

Logs and metrics will be saved in the `./logs` directory by default.
