import os
import random
from typing import Dict, Union

import numpy as np
# import tensorflow as tf

from sklearn.metrics import accuracy_score, roc_auc_score

def set_seed(seed: int = 42):
    # tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def cal_auc_score(model, data, feature_cols, label_col):
    pred_proba = model.predict_proba(data[feature_cols])
    if data[label_col].nunique() == 2:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba[:, 1])
    else:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba, multi_class="ovr")
    return auc

def cal_acc_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    acc = accuracy_score(data[label_col], pred)
    return acc

def cal_metrics(model, data, feature_cols, label_col):
    acc = cal_acc_score(model, data, feature_cols, label_col)
    auc = cal_auc_score(model, data, feature_cols, label_col)
    return {"ACC": acc, "AUC": auc}


