class OtherMatrix:
    def __init__(self):
        print("")

    def accuracy(self, cm, thres_prob, y_predicted, y_true):
        predictions = y_predicted > thres_prob

        # reset cunfusion matrix
        cm.reset()
        # compute confusion matrix
        cm.add(predictions, y_true)
        
        # accuracy 
        acc = (cm.TP() + cm.TN())/(cm.TP() + cm.FP() + cm.FN() + cm.TN())
        
        return acc

    def precision(self, cm, thres_prob, y_predicted, y_true):
        predictions = y_predicted > thres_prob

        # reset cunfusion matrix
        cm.reset()
        # compute confusion matrix
        cm.add(predictions, y_true)
        
        # precision
        pre = cm.TP()/(cm.TP() + cm.FP())
        
        return pre               

def recall(self, cm, thres_prob, y_predicted, y_true):
    predictions = y_predicted > thres_prob

    # reset cunfusion matrix
    cm.reset()
    # compute confusion matrix
    cm.add(predictions, y_true)
    
    # recall
    rec = cm.TP()/(cm.TP() + cm.FN())
    
    return rec
            