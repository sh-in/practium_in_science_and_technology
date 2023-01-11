import torch
import numpy as np

class EarlyStopping:
    def __init__(self, name, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc = 0
        self.train_acc = 0
        self.val_loss = np.Inf
        self.train_loss = np.Inf
        self.epoch = 0
        self.path = "checkpoint_model" + name + ".pth"
    
    def __call__(self, train_acc, train_loss, val_acc, val_loss, epoch, model):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(train_acc, train_loss, val_acc, val_loss, epoch, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(train_acc, train_loss, val_acc, val_loss, epoch, model)
            self.counter = 0

    def checkpoint(self, train_acc, train_loss, val_acc, val_loss, epoch, model):
        if self.verbose:
            print(f"Validation accuracy increased ({self.val_acc:.6f} --> {val_acc:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.epoch = epoch