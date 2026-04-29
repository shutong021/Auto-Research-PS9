# Program.md - Autonomous Fashion-MNIST Research

## 1. Research Problem and Scope

**Research Question**:  
Can we autonomously discover a better small CNN architecture and training configuration for Fashion-MNIST that improves validation accuracy while keeping the model simple and training fast?

**Single Optimization Metric**:  
`val_accuracy` (validation accuracy after training, higher is better)

**Fixed Resource Budget**:
- Maximum 180 seconds (3 minutes) wall-clock time per experiment
- Use Colab free T4 GPU
- Batch size fixed at 512
- Maximum 20 epochs per run
- No more than 500,000 parameters

**Simplicity Criterion**:  
Improvements smaller than 0.3% accuracy must not add complexity (layers, parameters, or code length).

## 2. File Structure and Boundaries

**Read-only files (DO NOT modify)**:
- `prepare.py` → data loading, model evaluation, logging
- `results.tsv` → experiment history (append only)

**Editable file (the ONLY file you may modify)**:
- `train.py` → model architecture, optimizer, learning rate, augmentations, etc.

**Strict Rules**:
- Never modify `prepare.py`
- Never add new dependencies (only use torch, torchvision, numpy, pandas)
- Never change the evaluation code or metric calculation
- Keep total parameters under 500k

## 3. The Experiment Loop

You will run in an infinite loop until the time budget for the whole project is reached. Each iteration follows these steps exactly:

1. **Idea Generation**  
   Review the last 5 experiments in results.tsv. Propose ONE small, focused change to `train.py` that could improve val_accuracy while respecting simplicity.

2. **Implementation**  
   Modify ONLY `train.py`. Save the file.

3. **Execution**  
   Run: `python train.py`

4. **Result Collection**  
   Read the printed validation accuracy and training time. Append a new row to `results.tsv`.

5. **Keep / Discard Decision**  
   - If new val_accuracy > best_so_far AND satisfies simplicity criterion → KEEP the change  
   - Else → Revert `train.py` to the previous best version

6. **Logging**  
   Always log the experiment with full details.

**Never stop** unless the human tells you to stop.

## 4. Output Format for results.tsv

Columns (tab-separated):
experiment_id | timestamp | idea | val_accuracy | train_time_sec | param_count | best_so_far | kept | notes

## 5. Human Oversight (Stop-and-Check Points)

Although the loop is autonomous, you MUST pause and ask the human at these points:
- After every 8 experiments
- When you reach a new best accuracy
- When you get stuck in the same idea for 3+ iterations
- If any experiment crashes twice in a row

---
