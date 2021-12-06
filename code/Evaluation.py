# coding: utf-8
# author: wcc
# create date: 2021-01-10 20:35
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np


class Evaluation:

    def prediction(self, real_score, pred_score):   # input np.array

        pred_label = []
        for i in pred_score:
            if i in sorted(pred_score)[-real_score.sum():]:
                pred_label.append(1)
            else:
                pred_label.append(0)

        pred_label = np.array(pred_label).astype(int)
        # print('real_score', real_score)
        # print('pred_score', pred_score)
        # print('pred_label', pred_label)

        # pred_label = pred_score > pred_score.mean()

        acc = (pred_label == real_score).mean()
        ap = average_precision_score(real_score, pred_score)
        f1 = f1_score(real_score, pred_label, average='macro')
        auc = roc_auc_score(real_score, pred_score)

        # print('micro', f1_score(real_score, pred_label, average='micro'))
        # print('macro', f1_score(real_score, pred_label, average='macro'))

        return acc, ap, f1, auc









