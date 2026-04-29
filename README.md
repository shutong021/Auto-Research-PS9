# PS9 - Autonomous Research Workflow (Fashion-MNIST)

This repository implements an autonomous research workflow powered by LLM agents, following Karpathy's autoresearch paradigm.

## Project Goal
Autonomously discover better small CNN architectures and training configurations on Fashion-MNIST to maximize validation accuracy under tight constraints.

## Files
- `program.md` → Main instruction for the AI agent (research org code)
- `prepare.py` → Read-only (data + evaluation)
- `train.py` → The only file the agent is allowed to modify
- `results.tsv` → Experiment history (auto-generated)

## How to Run
1. Open `PS9_FashionMNIST.ipynb` in Google Colab
2. Run the first cell to install dependencies and clone this repo
3. Let Gemini (or other agent) read `program.md` and start the autonomous loop

## Requirements
- Python 3.10+
- PyTorch + torchvision
- GPU (Colab T4)
