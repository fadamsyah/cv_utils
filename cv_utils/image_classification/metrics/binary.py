import torch

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

class BinaryBase:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, preds, target):
        y_true = target.cpu().detach().int()
        
        y_pred = torch.sigmoid(preds)
        y_pred = torch.where(y_pred >= self.threshold, 1, 0).cpu().detach().int()
        
        return y_pred.numpy(), y_true.numpy()

class Accuracy(BinaryBase):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
    
    def __call__(self, preds, target):
        y_pred, y_true = super().__call__(preds, target)
        return accuracy_score(y_true, y_pred)

class Precision(BinaryBase):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
    
    def __call__(self, preds, target):
        y_pred, y_true = super().__call__(preds, target)
        return precision_score(y_true, y_pred)

class Recall(BinaryBase):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
    
    def __call__(self, preds, target):
        y_pred, y_true = super().__call__(preds, target)
        return recall_score(y_true, y_pred)

class F1(BinaryBase):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
    
    def __call__(self, preds, target):
        y_pred, y_true = super().__call__(preds, target)
        return f1_score(y_true, y_pred)