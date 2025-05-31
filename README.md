# athens
an rl experiment in lean

## Lean sample project setup

In order to run the RL experiment we need to set up the dummy lean project.  Please install `lean` then:

``` sh
cd lean-project
lake update
```

## Python setup

Install `uv` and run `uv sync`.

## Pre-training

The pre-training script fine-tunes the language model on a dataset using a supervised training loop.

To run pre-training, use the `pre_training/lean_repository.py` script.
Basic parameters you can configure:
- epochs (default 10)
- learning rate (default 5e-4)
- warmup steps (default 1000)
- max training steps (default 100000)
- checkpoint directory (default ./checkpoints)
- tensorboard log directory (default ./logs)

Example command template (adjust as needed):

```sh
uv run python -m pre_training.lean_repository \
  --tokenizer_path checkpoints/tokenizer.json \
  --repo_path $PATH_TO_REPO_1  
```

The `repo_path` argument can be repeated.

The pre-training script will save checkpoints during training and log training and validation metrics for visualization with TensorBoard.

You can start TensorBoard with:

```sh
tensorboard --logdir=./logs
```

## Reinforcement learning

The reinforcement learning experiment fine-tunes the pretrained language model further with RL using DAPO loss.

Use the `reinforcement/run.py` script to launch the training.

Required arguments:
- `--init_checkpoint`: Path to the pretrained checkpoint from pre-training.
- `--tokenizer_path`: Path to the tokenizer JSON file.

Optional arguments:
- `--checkpoint`: Path to save checkpoints during RL fine-tuning.
- `--rounds`: Number of rounds of example generation and RL training (default: 3).
- `--epochs`: Number of epochs per round (default: 3).
- `--group_size`: Number of examples per group (default: 10).
- `--batch_size`: Batch size (default: 1).
- `--lr`: Learning rate (default: 5e-5).

Example command:

```sh
uv run python -m reinforcement.run \
  --init_checkpoint $INIT_CHECKPOINT \
  --tokenizer_path checkpoints/tokenizer.json \
  --checkpoint checkpoints/rl_recent.pt \
  --rounds 3 \
  --epochs 3 \
  --group_size 10 \
  --batch_size 1 \
  --lr 5e-5
```

This script will:
- Load the pretrained model and tokenizer.
- Repeatedly generate sample example groups and perform RL fine-tuning with DAPO loss on those groups.
- Save recent checkpoints to the specified path (if any).
- Log training progress to `reinforcement.log`.

---

# Reference

## Lean navigator dataset

Preprint: https://arxiv.org/abs/2503.04772  
Dataset: https://zenodo.org/records/13989482
