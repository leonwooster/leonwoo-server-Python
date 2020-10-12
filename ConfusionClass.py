import numpy as np

class ConfusionMatrix:
    def __init__(self):
        # init confusion matrix
        self.conf = np.ndarray((2, 2), np.int32)

    def reset(self):
        # reset to zero
        self.conf.fill(0)

    def add(self, pred, target):
        """
        This will take predicted probability and True label and compute confusion matrix
        """
        replace_indices = np.vstack((target.flatten(), pred.flatten())).T

        conf, _ = np.histogramdd(replace_indices, bins=(2, 2), range=[(0, 2), (0, 2)])

        self.conf += conf.astype(np.int32)

    def TP(self):
        return self.conf[1,1]
    
    def FP(self):
        return self.conf[0, 1]
    
    def TN(self):
        return self.conf[0, 0]
    
    def FN(self):
        return self.conf[1, 0]
    
    def confusion_matrix(self):
        """
        get confusion matrix as defined in figure
        """
        cm = np.array([[self.TP(), self.FP()],
                      [self.FN(), self.TN()]])
        return cm