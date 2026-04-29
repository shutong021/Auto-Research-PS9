# PS9 - Autonomous Research Workflow (Fashion-MNIST)

This repository implements an autonomous research workflow powered by LLM agents, inspired by Andrej Karpathy's autoresearch project.

## Project Goal
Autonomously discover better small CNN architectures and training configurations on Fashion-MNIST to maximize validation accuracy under tight constraints (time, parameters, and simplicity).

## Key Files
- `program.md` — Main research instructions and experiment loop (research org code)
- `prepare.py` — Read-only infrastructure (data loading + evaluation)
- `train.py` — The only file the agent is allowed to modify
- `results.tsv` — Full history of experiments
- `requirements.txt` — Python dependencies

## Best Result
**Validation Accuracy: 92.61%** (achieved with CosineAnnealingLR scheduler)

## How to Run
1. Open Google Colab and create a new notebook
2. Clone this repository:
   ```bash
   !git clone https://github.com/shutong021/Auto-Research-PS9.git
   %cd Auto-Research-PS9
3. Install dependencies: !pip install -r requirements.txt
4. Let Gemini (or other coding agent) read program.md and start the autonomous loop

## Requirements
Python 3.10+
PyTorch + torchvision
GPU (Colab T4 recommended)

**DOTE 6635 Artificial Intelligence for Business Research
**Spring 2026
