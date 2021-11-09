import torchmetrics

class Accuracy(torchmetrics.Accuracy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, preds, target, *args):
        return super().__call__(preds, target.int(), *args)

class Precision(torchmetrics.Precision):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, preds, target, *args):
        return super().__call__(preds, target.int(), *args)

class Recall(torchmetrics.Recall):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, preds, target, *args):
        return super().__call__(preds, target.int(), *args)

class F1(torchmetrics.F1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, preds, target, *args):
        return super().__call__(preds, target.int(), *args)
