from .utils import set_seed

from copy import deepcopy
from typing import List

import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y


class BaseClassifier:
    def __init__(
        self, input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study=False, pre_model=None
    ) -> None:
        self.ruleset: RuleSet = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config
        _, counts = np.unique(init_y, return_counts=True)
        self.class_ratio = counts.min() / counts.max()
        self.classes_ = unique_labels(init_y)
        self.onehoter = onehoter
        self.verbose = verbose
        self.pre_study = pre_study
        self.pre_model = pre_model

    def fit(self, X, y, eval_set):
        raise NotImplementedError()

    def predict_proba(self, X):
        return self.ruleset.predict_proba(X.values)

    def predict(self, X):
        return self.ruleset.predict(X.values)

    def evaluate(self, X, y):
        y_pred = self.predict(X)

        results = {}
        results["ACC"] = accuracy_score(y, y_pred)
        if self.output_dim == 2:
            results["AUC"] = roc_auc_score(y, y_pred)
            results["Precision"] = precision_score(y, y_pred, zero_division=0)
            results["Recall"] = recall_score(y, y_pred)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred)
            results["F1"] = f1_score(y, y_pred, zero_division=0)
        else:
            y_score = self.predict_proba(X)
            results["AUC"] = roc_auc_score(y, y_score, multi_class="ovr")
            results["Precision"] = precision_score(y, y_pred, average="macro", zero_division=0)
            results["Recall"] = recall_score(y, y_pred, average="macro", zero_division=0)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred, average="macro", zero_division=0)
            results["F1"] = f1_score(y, y_pred, average="macro", zero_division=0)
        return results
    
    def get_booster(self):
        raise NotImplementedError()


class XGBoostClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.output_dim == 2:
            self.model_config["objective"] = "binary:logitraw"
        else:
            self.model_config["objective"] = "multi:softmax"

        self.xgb = xgb.XGBClassifier(
            **self.model_config,
            num_class=self.output_dim if self.output_dim > 2 else None,
            eval_metric="auc",
            early_stopping_rounds=10,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = check_X_y(*eval_set)

        self.xgb.fit(X, y, eval_set=[eval_set], verbose=False)

    def predict_proba(self, X):
        return self.xgb.predict_proba(X)

    def predict(self, X):
        return self.xgb.predict(X)
    
    def get_booster(self):
        return self.xgb.get_booster()

def get_classifier(
    name,
    *,
    input_dim,
    output_dim,
    model_config,
    init_y,
    onehoter,
    pre_study=False,
    pre_model=None,
    seed=42,
    verbose=0,
):
    set_seed(seed=seed)

    if name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose)
    else:
        raise KeyError(f"{name} is not defined.")