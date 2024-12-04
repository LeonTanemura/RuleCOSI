import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from .utils import feature_name_combiner, label_encode

logger = logging.getLogger(__name__)


class TabularDataFrame(object):

    def __init__(
        self,
        seed,
        data_dir,
        categorical_encoder="ordinal",
        continuous_encoder: str = None,
        test_size: float = 0.2,
        **kwargs,
    ) -> None:
        self.seed = seed
        self.root = data_dir
        self.categorical_encoder = categorical_encoder
        self.continuous_encoder = continuous_encoder
        self.test_size = test_size

    def _init_checker(self):
        variables = [
            "continuous_columns",
            "categorical_columns",
            "target_column",
            "data",
        ]
        for variable in variables:
            if not hasattr(self, variable):
                if variable == "data":
                    if not (hasattr(self, "train") and hasattr(self, "test")):
                        raise ValueError(
                            "TabularDataFrame does not define `data`, but neither does `train`, `test`."
                        )
                else:
                    raise ValueError(
                        f"TabularDataFrame does not define a attribute: `{variable}`"
                    )

    def show_data_details(self, train: pd.DataFrame, test: pd.DataFrame):
        all_data = pd.concat([train, test])
        logger.info(f"Dataset size       : {len(all_data)}")
        logger.info(f"All columns        : {all_data.shape[1] - 1}")
        logger.info(f"Num of cate columns: {len(self.categorical_columns)}")
        logger.info(f"Num of cont columns: {len(self.continuous_columns)}")

        # 特徴量名を表示
        feature_columns = [col for col in all_data.columns if col != self.target_column]
        logger.info("Feature columns    :")
        for col in feature_columns:
            logger.info(f"  - {col}")

        y = all_data[self.target_column]
        class_ratios = y.value_counts(normalize=True)
        for label, class_ratio in zip(class_ratios.index, class_ratios.values):
            logger.info(f"class {label:<13}: {class_ratio:.3f}")

    def get_classify_dataframe(self) -> Dict[str, pd.DataFrame]:
        if hasattr(self, "train") and hasattr(self, "test"):
            train = self.train
            test = self.test
            all_target = label_encode(
                pd.concat([train[self.target_column], test[self.target_column]])
            )
            train[self.target_column] = all_target[: len(train)]
            test[self.target_column] = all_target[len(train) :]
            self.data = pd.concat([train, test])
        else:
            self.data[self.target_column] = label_encode(self.data[self.target_column])
            train, test = train_test_split(
                self.data,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=self.data[self.target_column],
            )

        self.show_data_details(train, test)
        classify_dfs = {
            "train": train,
            "test": test,
        }
        return classify_dfs

    def fit_feature_encoder(self, df_train):
        # Categorical values are fitted on all data.
        if self.categorical_columns != []:
            if self.categorical_encoder == "ordinal":
                self._categorical_encoder = OrdinalEncoder(dtype=np.int32).fit(
                    self.data[self.categorical_columns]
                )
            elif self.categorical_encoder == "onehot":
                self._categorical_encoder = OneHotEncoder(
                    sparse_output=False,
                    feature_name_combiner=feature_name_combiner,
                    dtype=np.int32,
                ).fit(self.data[self.categorical_columns])
            else:
                raise ValueError(self.categorical_encoder)
        if self.continuous_columns != [] and self.continuous_encoder is not None:
            if self.continuous_encoder == "standard":
                self._continuous_encoder = StandardScaler()
            elif self.continuous_encoder == "minmax":
                self._continuous_encoder = MinMaxScaler()
            else:
                raise ValueError(self.continuous_encoder)
            self._continuous_encoder.fit(df_train[self.continuous_columns])

    def apply_onehot_encoding(self, df: pd.DataFrame):
        encoded = self._categorical_encoder.transform(df[self.categorical_columns])
        encoded_columns = self._categorical_encoder.get_feature_names_out(
            self.categorical_columns
        )
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
        df = df.drop(self.categorical_columns, axis=1)
        return pd.concat([df, encoded_df], axis=1)

    def apply_feature_encoding(self, dfs):
        for key in dfs.keys():
            if self.categorical_columns != []:
                if isinstance(self._categorical_encoder, OrdinalEncoder):
                    dfs[key][self.categorical_columns] = (
                        self._categorical_encoder.transform(
                            dfs[key][self.categorical_columns]
                        )
                    )
                else:
                    dfs[key] = self.apply_onehot_encoding(dfs[key])
            if self.continuous_columns != []:
                if self.continuous_encoder is not None:
                    dfs[key][self.continuous_columns] = (
                        self._continuous_encoder.transform(
                            dfs[key][self.continuous_columns]
                        )
                    )
                else:
                    dfs[key][self.continuous_columns] = dfs[key][
                        self.continuous_columns
                    ].astype(np.float64)
        if self.categorical_columns != []:
            if isinstance(self._categorical_encoder, OneHotEncoder):
                self.categorical_columns = (
                    self._categorical_encoder.get_feature_names_out(
                        self.categorical_columns
                    )
                )
        return dfs

    def processed_dataframes(self) -> Dict[str, pd.DataFrame]:
        self._init_checker()
        dfs = self.get_classify_dataframe()
        # preprocessing
        self.fit_feature_encoder(dfs["train"])
        dfs = self.apply_feature_encoding(dfs)
        self.all_columns = list(self.categorical_columns) + list(
            self.continuous_columns
        )
        return dfs
