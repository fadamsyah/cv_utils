import torchmetrics

class Accuracy(torchmetrics.Accuracy):
    def __init__(self, use_cuda=False, **kwargs):
        super().__init__(**kwargs)
        self.use_cuda = use_cuda
    
    def __call__(self, preds, target, *args):
        y_true = target.int()
        if self.use_cuda:
            y_true = y_true.cuda()
        return super().__call__(preds, y_true, *args)

class Precision(torchmetrics.Precision):
    def __init__(self, use_cuda=False, **kwargs):
        super().__init__(**kwargs)
        self.use_cuda = use_cuda
    
    def __call__(self, preds, target, *args):
        y_true = target.int()
        if self.use_cuda:
            y_true = y_true.cuda()
        return super().__call__(preds, y_true, *args)

class Recall(torchmetrics.Recall):
    def __init__(self, use_cuda=False, **kwargs):
        super().__init__(**kwargs)
        self.use_cuda = use_cuda
    
    def __call__(self, preds, target, *args):
        y_true = target.int()
        if self.use_cuda:
            y_true = y_true.cuda()
        return super().__call__(preds, y_true, *args)

class F1(torchmetrics.F1):
    def __init__(self, use_cuda=False, **kwargs):
        super().__init__(**kwargs)
        self.use_cuda = use_cuda
    
    def __call__(self, preds, target, *args):
        y_true = target.int()
        if self.use_cuda:
            y_true = y_true.cuda()
        return super().__call__(preds, y_true, *args)