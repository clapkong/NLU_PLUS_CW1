# RNN & GRU — NLP Coursework

**University of Edinburgh** · Natural Language Understanding, Generation and Machine Translation (NLU+) · Spring 2025 (exchange)

NumPy-only implementation of Recurrent Neural Networks and Gated Recurrent Units, including manual gradient derivation and Backpropagation Through Time (BPTT). Achieved **full marks (30/30)** on automated evaluation.

---

## Assignment Overview

The coursework required implementing recurrent architectures at the mathematical level — no deep learning frameworks allowed. The goal was to demonstrate understanding of the forward/backward pass mechanics in sequence models, and to apply them to two NLP tasks:

1. **Language Modeling**: train an RNN to predict the next word in a sequence (Wikipedia corpus, vocab size 2000, cross-entropy objective).
2. **Number Agreement Prediction**: given a sentence containing a relative clause, predict subject-verb number agreement (binary: singular VBZ vs. plural VBP). This task probes whether recurrent models can track long-range syntactic dependencies — a classic evaluation setting from Linzen et al. (2016).

---

## Implementation

The model abstractions, training loop, GRU backward pass, and test suite were provided as skeleton code. Core implementations:

- **RNN** (`rnn.py`): forward pass, standard backpropagation, and truncated BPTT — all gradient accumulation written from scratch
- **GRU** (`gru.py`): forward cell (reset/update gates, candidate state), with gradient dispatch to the provided abstract backward pass
- **Training utilities** (`runner.py`): cross-entropy loss for LM, loss/accuracy for number agreement, training mode setup for all three configurations

---

## Results

| Component | Points |
|-----------|--------|
| RNN forward pass | 5/5 |
| RNN loss computation | 5/5 |
| RNN standard BP | 5/5 |
| RNN BPTT | 5/5 |
| Number prediction — RNN (BP + BPTT) | 5/5 |
| Number prediction — GRU (BP + BPTT) | 5/5 |
| **Total** | **30/30** |

---

## Written Report

### Assignment
**Q2**: Tune hyperparameters (learning rate × hidden units × BPTT steps); train best model on 25K sentences; report test loss and perplexity.

**Q3**: Compare RNN and GRU across hidden unit sizes (10, 25, 50) on the number agreement task.

**Q4**: Design one experiment comparing RNN vs. GRU under varying BPTT steps, with a hypothesis and interpretation.


### Report (`report.pdf`)
**Q2**: Ran all 18 hyperparameter configurations (lr ∈ {0.05, 0.1, 0.5} × hidden ∈ {25, 50} × BPTT ∈ {0, 2, 5}), recording training time per run alongside loss. Final model trained on 25K sentences; test perplexity: **86.78** (dev loss 4.459, test loss 4.463).

**Q3**: Compared RNN and GRU with hidden units (10, 25, 50), tracking loss and accuracy at every epoch. Best accuracy: GRU-50 at **75.1%** vs. RNN-50 at **68.7%**. Analysed gradient stability differences and included formal computational complexity derivation — RNN O(T(s²+sd)) vs. GRU O(T(3s²+3sd)) — in an appendix.

**Q4**: Ran two experiments where only one was required.

  1. **BPTT step analysis**: Tested BPTT ∈ {5, 10, 20, 30, 50} over 40 epochs on 25K sentences. Extended the training loop to log gradient norms (∥ΔU∥ for RNN; ∥ΔUr∥, ∥ΔUz∥, ∥ΔUh∥ for GRU) at every timestep. Finding: BPTT=10 optimal; performance plateaus beyond ~15 steps.
  2. **Gate ablation study** (self-designed): Trained four GRU variants — baseline, reset-gate disabled, update-gate disabled, both disabled. Finding: disabling the update gate hurts more than disabling the reset gate; both-disabled underperformed even a standard RNN despite having more parameters.

---

## Usage

```bash
pip install -r code/requirements.txt

# Language model
python code/runner.py train-lm-rnn data/ <hidden_dim> <bptt_steps> <lr>

# Number prediction — RNN
python code/runner.py train-np-rnn data/ <hidden_dim> <bptt_steps> <lr>

# Number prediction — GRU
python code/runner.py train-np-gru data/ <hidden_dim> <bptt_steps> <lr>

# Example
python code/runner.py train-np-gru data/ 50 3 0.5
```

---

## Stack

Python 3 · NumPy · Wikipedia (train / dev / test splits)
