import logging
import os
from time import time

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold

import dataset
from dataset import TabularDataFrame
from .classifier import get_classifier
from .utils import (
    cal_metrics,
    set_seed,
)

logger = logging.getLogger(__name__)


class ExpBase:
    def __init__(self, config):
        set_seed(config.seed)

        self.n_splits = config.n_splits
        self.model_name = config.model.name

        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        dataframe: TabularDataFrame = getattr(dataset, self.data_config.name)(
            seed=config.seed, **self.data_config
        )
        dfs = dataframe.processed_dataframes()
        self.train, self.test = dfs["train"], dfs["test"]
        self.columns = dataframe.all_columns
        self.target_column = dataframe.target_column

        self.seed = config.seed
        self.init_writer()

    def init_writer(self):
        metrics = [
            "fold",
            "ACC",
            "AUC",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
        ]
        self.writer = {m: [] for m in metrics}

    def add_results(self, i_fold, scores: dict, time):
        self.writer["fold"].append(i_fold)
        for m in self.writer.keys():
            if m == "fold":
                continue
            self.writer[m].append(scores[m])

    def each_fold(self, i_fold, train_data, val_data):
        uniq = self.get_unique(train_data)
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(
            i_fold=i_fold, x=x, y=y, val_data=val_data, uniq=uniq
        )
        model = get_classifier(
            self.model_name,
            input_dim=len(self.columns),
            output_dim=len(uniq),
            model_config=model_config,
            init_y=y,
            onehoter=None,
            verbose=self.exp_config.verbose,
            seed=self.seed,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=(
                val_data[self.columns],
                val_data[self.target_column].values.squeeze(),
            ),
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end

    def run(self):
        skf = StratifiedKFold(n_splits=self.n_splits)
        for i_fold, (train_idx, val_idx) in enumerate(
            skf.split(self.train, self.train[self.target_column])
        ):
            if len(self.writer["fold"]) != 0 and self.writer["fold"][-1] >= i_fold:
                logger.info(f"Skip {i_fold + 1} fold. Already finished.")
                continue

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            print("rule作成成功")
            exit()  # RuleCOSI+のアルゴリズムを作成しないと以下を実行できない。

            score = cal_metrics(model, val_data, self.columns, self.target_column)
            score.update(
                model.evaluate(
                    val_data[self.columns],
                    val_data[self.target_column].values.squeeze(),
                )
            )
            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] val/ACC: {score['ACC']:.4f} | val/AUC: {score['AUC']:.4f} | "
            )

            score = cal_metrics(model, self.test, self.columns, self.target_column)
            score.update(
                model.evaluate(
                    self.test[self.columns],
                    self.test[self.target_column].values.squeeze(),
                )
            )

            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] test/ACC: {score['ACC']:.4f} | test/AUC: {score['AUC']:.4f} | "
            )
            self.add_results(i_fold, score, time)

        logger.info(f"[{self.model_name} Test Results]")
        mean_std_score = {}
        score_list_dict = {}
        for k, score_list in self.writer.items():
            if k == "fold":
                continue
            score = np.array(score_list)
            mean_std_score[k] = f"{score.mean(): .4f} ±{score.std(ddof=1): .4f}"
            score_list_dict[k] = score_list
            logger.info(f"[{self.model_name} {k}]: {mean_std_score[k]}")

    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()

    def get_unique(self, train_data):
        uniq = np.unique(train_data[self.target_column])
        return uniq

    def get_x_y(self, train_data):
        x, y = train_data[self.columns], train_data[self.target_column].values.squeeze()
        return x, y


class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)

    def get_model_config(self, *args, **kwargs):
        return self.model_config
