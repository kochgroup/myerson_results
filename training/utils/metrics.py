import torchmetrics
def binary_auroc(preds, target, *args, **kwargs):
    # auroc expects long/int labels
    target = target.long()
    return torchmetrics.functional.classification.binary_auroc(preds, target, *args, **kwargs)

class ModBinaryF1Score(torchmetrics.classification.BinaryF1Score):
    def __init__(self):
        super().__init__()

    def update(self, preds, target):
        super().update(preds, target.long())

class ModBinaryAUROC(torchmetrics.classification.BinaryAUROC):
    def __init__(self):
        super().__init__()

    def update(self, preds, target):
        super().update(preds, target.long())

class ModBinaryAccuracy(torchmetrics.classification.BinaryAccuracy):
    def __init__(self):
        super().__init__()

    def update(self, preds, target):
        super().update(preds, target.long())

class ModKendallRankCorrCoef(torchmetrics.KendallRankCorrCoef):
    def __init__(self):
        super().__init__()

    def update(self, preds, target):
        super().update(preds.squeeze(), target.squeeze())
