# PS9 - Autonomous Research Workflow (Fashion-MNIST)

This repository implements an autonomous research workflow powered by LLM agents, inspired by Andrej Karpathy's autoresearch project.

## Project Goal
Autonomously discover better small CNN architectures and training configurations on Fashion-MNIST to maximize validation accuracy under tight constraints (time, parameters, and simplicity).

## Key Files
- `program.md` — Main research instructions and experiment loop (the "prompt").
- `prepare.py` — Read-only infrastructure (data loading + evaluation).
- `train.py` — The only file the AI agent is allowed to modify.
- `results.tsv` — Full history of experiments.
- `gemini_execution_transcript.txt` — The full, unedited chat transcript showing the AI agent's reasoning and code generation.
- `PS9_FashionMNIST.ipynb` —  A Google Colab notebook for easy reproduction.
- `requirements.txt` — Python dependencies.

## Best Result
**Validation Accuracy: 92.61%** (achieved with CosineAnnealingLR scheduler)

## How to Run
### Option 1: The Easy Way (Google Colab - Recommended)
For easy reproducibility without local environment setup, please use the provided Colab notebook:
1. Open `PS9_FashionMNIST.ipynb` in Google Colab.
2. Run the first cell to clone this repository and install necessary dependencies.
3. Run the subsequent cells to execute `!python train.py` and observe the training process.

### Option 2: Local Execution
If you prefer to run it locally or on your own server:
```bash
git clone https://github.com/shutong021/Auto-Research-PS9.git
cd Auto-Research-PS9
pip install -r requirements.txt
python train.py
```

## How the Autonomous Loop Works
Instead of coding the model manually, the human orchestrator interacts with an LLM agent (Gemini) using the rules defined in program.md:

The agent reads train.py and past results in results.tsv.

It proposes a focused architectural or hyperparameter change and explicitly states its reasoning.

The human copies the updated code into train.py, runs the script, and feeds the terminal output back to the agent.

The agent evaluates the results against the simplicity criterion defined in program.md and decides whether to KEEP or REJECT the changes.

## Requirements
Python 3.10+
PyTorch + torchvision
GPU (Colab T4 recommended)

##### DOTE 6635 Artificial Intelligence for Business Research
##### Spring 2026
