import torchmetrics
from torch.nn import functional as F

class Accuracy(torchmetrics.Accuracy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, preds, target, *args):
        y_pred = F.sigmoid(preds)
        y_pred = y_pred.cpu()
        y_true = target.cpu().int()
        return super().__call__(y_pred, y_true, *args)

class Precision(torchmetrics.Precision):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, preds, target, *args):
        y_pred = F.sigmoid(preds)
        y_pred = y_pred.cpu()
        y_true = target.cpu().int()
        return super().__call__(y_pred, y_true, *args)

class Recall(torchmetrics.Recall):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, preds, target, *args):
        y_pred = F.sigmoid(preds)
        y_pred = y_pred.cpu()
        y_true = target.cpu().int()
        return super().__call__(y_pred, y_true, *args)

class F1(torchmetrics.F1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, preds, target, *args):
        y_pred = F.sigmoid(preds)
        y_pred = y_pred.cpu()
        y_true = target.cpu().int()
        return super().__call__(y_pred, y_true, *args)