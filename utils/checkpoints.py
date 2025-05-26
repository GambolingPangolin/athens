#!/usr/bin/env python3


import os
import torch


class CheckpointManager:
    def __init__(self, ckpt_dir, max_train=2, max_val=1):
        """
        Manage saving checkpoints with top best for train and val losses.

        Args:
            ckpt_dir (str): directory to save checkpoints
            max_train (int): number of best train loss checkpoints to keep
            max_val (int): number of best val loss checkpoints to keep
        """
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.max_train = max_train
        self.max_val = max_val

        # Lists of tuples: (loss, path)
        self.best_train_ckpts = []
        self.best_val_ckpts = []

        self.recent_ckpt_path = os.path.join(self.ckpt_dir, "recent_checkpoint.pt")

    def _save_ckpt(self, state, prefix, loss, epoch):
        filename = f"{prefix}_best_epoch{epoch+1}_loss{loss:.6f}.pt"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, path)
        return path

    def _update_ckpts(self, ckpt_list, max_keep, loss, path):
        # Insert sorted ascending (lowest loss first)
        ckpt_list.append((loss, path))
        ckpt_list.sort(key=lambda x: x[0])
        # Remove extra
        while len(ckpt_list) > max_keep:
            _, to_rm = ckpt_list.pop(-1)
            if os.path.exists(to_rm):
                os.remove(to_rm)

    def register(
        self, model, optimizer, scheduler, epoch, train_loss=None, val_loss=None
    ):
        """
        Save checkpoints if they are among the best according to train/val losses.

        Args:
            model (torch.nn.Module)
            optimizer (torch.optim.Optimizer)
            scheduler (torch.optim.lr_scheduler._LRScheduler)
            epoch (int)
            train_loss (float or None)
            val_loss (float or None)
        """
        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        # Always save recent checkpoint (overwrite)
        torch.save(state, self.recent_ckpt_path)

        saved = False

        if val_loss is not None:
            # Save if val_loss belongs in top max_val
            if (
                len(self.best_val_ckpts) < self.max_val
                or val_loss < self.best_val_ckpts[-1][0]
            ):
                path = self._save_ckpt(state, "val", val_loss, epoch)
                self._update_ckpts(self.best_val_ckpts, self.max_val, val_loss, path)
                saved = True

        if train_loss is not None:
            # Save if train_loss belongs in top max_train
            if (
                len(self.best_train_ckpts) < self.max_train
                or train_loss < self.best_train_ckpts[-1][0]
            ):
                path = self._save_ckpt(state, "train", train_loss, epoch)
                self._update_ckpts(
                    self.best_train_ckpts, self.max_train, train_loss, path
                )
                saved = True

        return saved
