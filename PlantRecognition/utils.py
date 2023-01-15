import numpy as np
import os
import torch
import random


def seed_everything(seed):
    """
    Makes code deterministic using a given seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stops the training if the validation loss doesn't improve after given patience
    """

    def __init__(self, patience=7, verbose=False, delta=0.0001, path="checkpoint.pt"):
        """
        :param patience: How long to wait after lat time validation loss improved
        :param verbose: if True, prints a message for each validation loss improvement
        :param delta: Minimum change in the monitoring quantity to qualify as an improvement
        :param path: Path for the checkpoint to be saved to
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """ Saves model when validation loss decrease """
        if self.verbose:
            print(f"Validation loss decreased({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
