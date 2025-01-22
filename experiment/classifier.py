from .utils import set_seed

from copy import deepcopy
from typing import List

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier as SklearnRFClassifier

from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y

import rulecosi


class BaseClassifier:
    def __init__(
        self,
        input_dim,
        output_dim,
        model_config,
        init_y,
        onehoter,
        verbose,
        pre_study=False,
        pre_model=None,
    ) -> None:
        self.ruleset = None
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
            results["Precision"] = precision_score(
                y, y_pred, average="macro", zero_division=0
            )
            results["Recall"] = recall_score(
                y, y_pred, average="macro", zero_division=0
            )
            results["Specificity"] = recall_score(
                1 - y, 1 - y_pred, average="macro", zero_division=0
            )
            results["F1"] = f1_score(y, y_pred, average="macro", zero_division=0)
        return results

    def get_booster(self):
        raise NotImplementedError()


class RuleCOSIClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ensemble_config = self.model_config.ensemble
        self.ensemble_name = self.model_config.name
        rulecosi_config = self.model_config.rulecosi

        if self.output_dim == 2:
            ensemble_config["objective"] = "binary:logitraw"
        else:
            ensemble_config["objective"] = "multi:softmax"

        if self.pre_model is not None:
            self.ens = self.pre_model
        elif self.ensemble_name == "randomforest":
            if "objective" in ensemble_config:
                ensemble_config = dict(ensemble_config)  # DictConfigを普通の辞書に変換
                ensemble_config.pop("objective", None)  # 'objective'を削除

            self.ens = SklearnRFClassifier(
                **ensemble_config,
            )
        elif self.ensemble_name == "lightgbm":
            ensemble_config["objective"] = "binary"
            self.ens = lgb.LGBMClassifier(
                **ensemble_config,
                num_class=self.output_dim if self.output_dim > 2 else None,
                eval_metric="auc",
                early_stopping_rounds=10,
            )
        else:
            self.ens = xgb.XGBClassifier(
                **ensemble_config,
                num_class=self.output_dim if self.output_dim > 2 else None,
                eval_metric="auc",
                early_stopping_rounds=10,
            )
        self.rulecosi = rulecosi.RuleCOSIClassifier(
            base_ensemble=self.ens,
            metric="auc",
            **rulecosi_config,
            verbose=self.verbose,
        )

    def pre_fit(self, X, y, eval_set):
        X_xgb, y_xgb = check_X_y(X, y)
        eval_set_xgb = check_X_y(*eval_set)
        self.pre_model = self.ens.fit(
            X_xgb, y_xgb, eval_set=[eval_set_xgb], verbose=False
        )

    def fit(self, X, y, eval_set):
        if self.ensemble_name == "randomforest":
            X_rf, y_rf = check_X_y(X, y)
            self.ens.fit(X_rf, y_rf)
        elif self.ensemble_name == "lightgbm":
            X_lgb, y_lgb = check_X_y(X, y)
            eval_set_lgb = check_X_y(*eval_set)
            self.ens.fit(X_lgb, y_lgb, eval_set=[eval_set_lgb])
        else:
            X_xgb, y_xgb = check_X_y(X, y)
            eval_set_xgb = check_X_y(*eval_set)
            self.ens.fit(X_xgb, y_xgb, eval_set=[eval_set_xgb], verbose=False)
        self.rulecosi.fit(X, y)
        self.ruleset = self.rulecosi.simplified_ruleset_

    def predict_proba(self, X):
        return softmax(self.ruleset.predict_proba(X.values), axis=1)

    def predict(self, X):
        return self.rulecosi.predict(X.values)


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


class LightGBMClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.output_dim == 2:
            self.model_config["objective"] = "binary"
        else:
            self.model_config["objective"] = "multiclass"
            self.model_config["num_class"] = self.output_dim

        self.lgbm = lgb.LGBMClassifier(**self.model_config)

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set_X, eval_set_y = check_X_y(*eval_set)
        eval_set = [(eval_set_X, eval_set_y)]

        self.lgbm.fit(X, y, eval_set=eval_set, verbose=self.verbose)

    def predict_proba(self, X):
        return self.lgbm.predict_proba(X)

    def predict(self, X):
        return self.lgbm.predict(X)


class RandomForestClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.rf = SklearnRFClassifier(**self.model_config)

    def fit(self, X, y, eval_set=None):
        # eval_set is ignored for RandomForest
        self._column_names = X.columns
        X, y = check_X_y(X, y)

        self.rf.fit(X, y)

    def predict_proba(self, X):
        return self.rf.predict_proba(X)

    def predict(self, X):
        return self.rf.predict(X)


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
        return XGBoostClassifier(
            input_dim, output_dim, model_config, init_y, onehoter, verbose
        )
    elif name == "rulecosi":
        return RuleCOSIClassifier(
            input_dim,
            output_dim,
            model_config,
            init_y,
            onehoter,
            verbose,
            pre_study,
            pre_model,
        )
    elif name == "lightgbm":
        return LightGBMClassifier(input_dim, output_dim, model_config, init_y, onehoter)
    elif name == "randomforest":
        return RandomForestClassifier(
            input_dim, output_dim, model_config, init_y, onehoter, verbose
        )
    else:
        raise KeyError(f"{name} is not defined.")
