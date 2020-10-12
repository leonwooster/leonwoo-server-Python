import torch
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (8, 8)

class ROCCurve:
    def __init__(self, y_test, y_pred_score):
        # Init attributs 
        self.y_test = y_test
        self.y_pred_score = y_pred_score

    def _get_fpr_tpr(self):
        # thresholds
        thresholds = torch.linspace(0.001, 0.999, 1000).unsqueeze(1)
        
        # get prediction for all thresholds
        self.y_pred = self.y_pred_score.T > thresholds
        
        # get TP, FP, TN, and FN for all thresholds
        tp, fp, tn, fn = self._get_tp_fp_tn_fn()
        
        # calculate true positive rate for all thresholds
        tpr = tp.float() / (tp + fn)
        
        # calculate false positive rate for all thresholds
        fpr = fp.float() / (fp + tn)
        
        return fpr.flip((0, )), tpr.flip((0, ))
        

    def _get_tp_fp_tn_fn(self):
        
        # change datatype to bool
        self.y_pred = self.y_pred.bool()
        self.y_test = self.y_test.bool()
        
        # calculate TP
        tp = (self.y_pred & self.y_test).sum(dim=1)
        
        # calculate FP
        fp = (self.y_pred & ~self.y_test).sum(dim=1)
        
        # calculate TN
        tn = (~self.y_pred & ~self.y_test).sum(dim=1)
        
        # calculate FN
        fn = (~self.y_pred & self.y_test).sum(dim=1)
        
        return tp, fp, tn, fn

    def plot_roc(self):
        
        # get TPR and FPR and plot TPR-vs-FPR
        plt.plot(*self._get_fpr_tpr(), label='ROC curve', color='g')
        plt.plot([0, 1], [0, 1], label='Random Classifier (AUC = 0.5)', linestyle='--', lw=2, color='r')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title('ROC Curve')
        plt.show()

    def get_auc_score(self):
        # Get TPR and FPR
        fpr, tpr = self._get_fpr_tpr()
        
        # get area under the curve of TPR-vs-FPR plot
        return np.trapz(tpr, fpr), fpr, tpr