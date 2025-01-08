import time
from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd

import scipy.stats as st

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from rule_making import RuleExtraction
from ._simplify_rulesets import _simplify_rulesets
from .combine import Combine
from .pruning import SCPruning
from .generalize import Generalize
from rule_making import RuleSet, Rule
from rule_making import RuleHeuristics
from .utils import sort_ruleset


def _ensemble_type(ensemble):
    """Return the ensemble type

    :param ensemble:
    :return:
    """
    if isinstance(ensemble, GradientBoostingClassifier):
        return "gbt"
    elif str(ensemble.__class__) == "<class 'xgboost.sklearn.XGBClassifier'>":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "If you want to use "
                "xgboost.sklearn.XGBClassifier "
                "ensembles you should install xgboost "
                "library."
            )
        return "gbt"
    else:
        raise NotImplementedError


class BaseRuleCOSI(BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for RuleCOSI estimators."""

    def __init__(
        self,
        base_ensemble=None,
        n_estimators=5,
        tree_max_depth=3,
        percent_training=None,
        early_stop=0,
        metric="f1",
        float_threshold=-1e-6,
        column_names=None,
        random_state=None,
        verbose=0,
    ):
        self.base_ensemble = base_ensemble
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.percent_training = percent_training
        self.early_stop = early_stop
        self.metric = metric
        self.float_threshold = float_threshold
        self.column_names = column_names
        self.random_state = random_state
        self.verbose = verbose

    def _more_tags(self):
        return {"binary_only": True}

    def fit(self, X, y):
        """Combine and simplify the decision trees from the base ensemble
        and builds a rule-based classifier using the training set (X,y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.



        Returns
        -------
        self : object
        """
        pass

    @abstractmethod
    def _validate_and_create_base_ensemble(self):
        """Validate the parameter of base ensemble and if it is None,
        it set the default ensemble GradientBoostingClassifier.

        """

        pass


class RuleCOSIClassifier(ClassifierMixin, BaseRuleCOSI):
    """Tree ensemble Rule COmbiantion and SImplification algorithm for
    classification

    RuleCOSI extract, combines and simplify rules from a variety of tree
    ensembles and then constructs a single rule-based model that can be used
    for classification [1]. The ensemble is simpler and have a similar
    classification performance compared than that of the original ensemble.
    Currently only accept binary classification (March 2021)

    Parameters
    ----------
    base_ensemble: BaseEnsemble object, default = None
        A BaseEnsemble estimator object. The supported types are:
       - :class:`sklearn.ensemble.RandomForestClassifier`
       - :class:`sklearn.ensemble.BaggingClassifier`
       - :class:`sklearn.ensemble.GradientBoostingClassifier`
       - :class:`xgboost.XGBClassifier`
       - :class:`catboost.CatBoostClassifier`
       - :class:`lightgbm.LGBMClassifier`

       If the estimator is already fitted, then the parameters n_estimators
       and max_depth used for fitting the ensemble are used for the combination
       process. If the estimator is not fitted, then the estimator will be
       first fitted using the provided parameters in the RuleCOSI object.
       Default value is None, which uses a
       :class:`sklearn.ensemble.GradientBoostingClassifier` ensemble.

    n_estimators: int, default=5
        The number of estimators used for fitting the ensemble,
        if it is not already fitted.

    tree_max_depth: int, default=3
        The maximum depth of the individual tree estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    cov_threshold: float, default=0.0
        Coverage threshold of a rule to be considered for further combinations.
        (beta in the paper)The greater the value the more rules are discarded.
        Default value is 0.0, which it only discards rules with null coverage.

    conf_threshold: float, default=0.5
        Confidence or rule accuracy threshold of a rule to be considered for
        further combinations (alpha in the paper). The greater the value, the
        more rules are discarded. Rules with high confidence are accurate rules.
        Default value is 0.5, which represents rules with higher than random
        guessing accuracy.

    c= float, 0.25
        Confidence level for estimating the upper bound of the statistical
        correction of the rule error. It is used for the generalization process.

    percent_training= float, default=None
        Percentage of the training used for the combination and simplification
        process. If None, all the training data is usded. This is useful when
        the training data is too big because it helps to accelerate the
        simplification process. (experimental)

    early_stop: float, default=0
        This parameter allows the algorithm to stop if a certain amount of
        iterations have passed without improving the metric. The amount is
        obtained from the truncated integer of n_estimators * ealry_stop.

    metric: string, default='f1'
        Metric that is optimized in the combination process. The default is
        f1. Other accepted measures are:
         - 'roc_auc' for AUC under the ROC curve
         - 'accuracy' for Accuracy

    rule_order: string, default 'supp'
        Defines the way in the rules are ordered on each iteration. 'cov' order
        the rules first by coverage and 'conf' order the rules first by
        confidence or rule accuracy. 'supp' orders the rule by their support.
        This parameter affects the combination process and can be chosen
        conveniently depending on the desired results.

    sort_by_class: bool or list, default None
        Define if the combined rules are ordered by class. If 'None', the rules
        are not forced to be ordered (although in some cases the result will
        be ordered). If 'True', the rules are ordered by class
        lexicographically. If a list is provided, the rules will be ordered by
        class following the order provided by the user. The list should
        contain the actual classes passed in y when calling the fit method.

    column_names: array of string, default=None
        Array of strings with the name of the columns in the data. This is
        useful for displaying the name of the features in the generated rules.

    random_state: int, RandomState instance or None, default=None
        Controls the random seed given to the ensembles when trained. RuleCOSI
        does not have any random process, so it affects only the ensemble
        training.

    verbose: int, default=0
        Controls the output of the algorithm during the combination process. It
         can have the following values:
        - 0 is silent
        - 1 output only the main stages of the algorithm
        - 2 output detailed information for each iteration

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    original_rulesets_ : array of RuleSet, shape (n_estimators,)
        The original rulesets extracted from the base ensemble.

    simplified_ruleset_ : RuleSet
        Combined and simplified ruleset extracted from the base ensemble.

    n_combinations_ : int
        Number of rule-level combinations performed by the algorithm.

    combination_time_ : float
        Time spent for the combination and simplification process

    ensemble_training_time_ : float
        Time spent for the ensemble training. If the ensemble was already
        trained, this is 0.


    References
    ----------
    .. [1] Obregon, J., Kim, A., & Jung, J. Y., "RuleCOSI: Combination and
           simplification of production rules from boosted decision trees for
           imbalanced classification", 2019.

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.datasets import make_classification
    >>> from rulecosi import RuleCOSIClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(),
    ...                          n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(),
                       n_estimators=100, random_state=0)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    >>> clf.score(X, y)
    0.966...

    """

    def __init__(
        self,
        base_ensemble=None,
        n_estimators=5,
        tree_max_depth=3,
        cov_threshold=0.0,
        conf_threshold=0.5,
        c=0.25,
        percent_training=None,
        early_stop=0,
        metric="f1",
        rule_order="supp",
        sort_by_class=None,
        float_threshold=1e-6,
        column_names=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            base_ensemble=base_ensemble,
            n_estimators=n_estimators,
            tree_max_depth=tree_max_depth,
            percent_training=percent_training,
            early_stop=early_stop,
            metric=metric,
            float_threshold=float_threshold,
            column_names=column_names,
            random_state=random_state,
            verbose=verbose,
        )

        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.c = c
        self.rule_order = rule_order
        self.sort_by_class = sort_by_class

    def fit(self, X, y):
        """Combine and simplify the decision trees from the base ensemble
        and builds a rule-based classifier using the training set (X,y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.



        Returns
        -------
        self : object
        """
        self._rule_extractor = None
        self._rule_heuristics = None
        self._base_ens_type = None
        self._weights = None
        self._global_condition_map = None
        self._bad_combinations = None
        self._good_combinations = None
        self._early_stop_cnt = 0
        self.alpha_half_ = None

        self.X_ = None
        self.y_ = None
        self.classes_ = None
        self.processed_rulesets_ = None
        self.simplified_ruleset_ = None
        self.combination_time_ = None
        self.n_combinations_ = None
        self.ensemble_training_time_ = None

        # Check that X and y have correct shape
        if self.column_names is None:
            if isinstance(X, pd.DataFrame):
                self.column_names = X.columns
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.alpha_half_ = st.norm.ppf(1 - (self.c / 2))

        if self.percent_training is None:
            self.X_ = X
            self.y_ = y
        else:
            x, _, y, _ = train_test_split(
                X,
                y,
                test_size=(1 - self.percent_training),
                shuffle=True,
                stratify=y,
                random_state=self.random_state,
            )
            self.X_ = x
            self.y_ = y

        # add rule ordering funcionality (2023.03.30)
        self._sorting_list = ["cov", "conf", "supp"]
        self._sorting_list.remove(self.rule_order)
        self._sorting_list.insert(0, self.rule_order)
        self._sorting_list.reverse()
        if self.sort_by_class is not None:
            if isinstance(self.sort_by_class, bool):
                self.sort_by_class = self.classes_.tolist()

        if self.n_estimators is None or self.n_estimators < 2:
            raise ValueError(
                "Parameter n_estimators should be at "
                "least 2 for using the RuleCOSI method."
            )

        if self.verbose > 0:
            print("Validating original ensemble...")
        try:
            if self.base_ensemble is None:
                self.base_ensemble_ = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state,
                )
            else:
                self.base_ensemble_ = self.base_ensemble
            self._base_ens_type = _ensemble_type(self.base_ensemble_)
        except NotImplementedError:
            print(
                f"Base ensemble of type {type(self.base_ensemble_).__name__} "
                f"is not supported."
            )
        try:
            check_is_fitted(self.base_ensemble_)
            self.ensemble_training_time_ = 0
            if self.verbose > 0:
                print(
                    f"{type(self.base_ensemble_).__name__} already trained, "
                    f"ignoring n_estimators and "
                    f"tree_max_depth parameters."
                )
        except NotFittedError:
            self.base_ensemble_ = self._validate_and_create_base_ensemble()
            if self.verbose > 0:
                print(
                    f"Training {type(self.base_ensemble_).__name__} "
                    f"base ensemble..."
                )
            start_time = time.time()
            self.base_ensemble_.fit(X, y)
            end_time = time.time()
            self.ensemble_training_time_ = end_time - start_time
            if self.verbose > 0:
                print(
                    f"Finish training {type(self.base_ensemble_).__name__} "
                    f"base ensemble"
                    f" in {self.ensemble_training_time_} seconds."
                )

        start_time = time.time()

        self._rule_extractor = RuleExtraction(
            base_ensemble_=self.base_ensemble_,
            column_names=self.column_names,
            classes_=self.classes_,
            X_=self.X_,
            y_=self.y_,
        )
        self.processed_rulesets_, self._global_condition_map, self._rule_heuristics = (
            self._rule_extractor.rule_extraction()
        )
        self.processed_rulesets_ = _simplify_rulesets(
            self.processed_rulesets_, self._global_condition_map
        )

        for ruleset in self.processed_rulesets_:
            sort_ruleset(ruleset)

        self.simplified_ruleset_ = self.processed_rulesets_[0]

        for ruleset in self.processed_rulesets_[1:]:
            # print(f"initial ruleset length: {len(ruleset.rules)}")
            self.combiner = self.class_maker("combine")
            self.simplified_ruleset_ = self.combiner.combine_rulesets(
                self.simplified_ruleset_, ruleset
            )
            # print(f"combined ruleset length: {len(self.simplified_ruleset_.rules)}")
            self.pruner = self.class_maker("pruning")
            self.simplified_ruleset_ = self.pruner.sequential_covering_pruning(
                self.simplified_ruleset_
            )
            # print(f"pruned ruleset length: {len(self.simplified_ruleset_.rules)}")
            self.generalizer = self.class_maker("generalize")
            self.simplified_ruleset_ = self.generalizer.generalize_ruleset(
                self.simplified_ruleset_
            )
            # print(f"generalized ruleset length: {len(self.simplified_ruleset_.rules)}")

        self.add_default_ruleset()

    def _more_tags(self):
        return {"binary_only": True}

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample. The prediction use the
        simplified ruleset and evaluate the rules one by one. When a rule
        covers a sample, the head of the rule is returned as predicted class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted class. The class with the highest value in the class
            distribution of the fired rule.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        if self.X_.shape[1] != X.shape[1]:
            raise ValueError(
                f"X contains {X.shape[1]} features, but RuleCOSIClassifier was"
                f" fitted with {self.X_.shape[1]}."
            )

        return self.simplified_ruleset_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is obtained from
        the class distribution of the fired rule.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        if self.X_.shape[1] != X.shape[1]:
            raise ValueError(
                f"X contains {X.shape[1]} features, but RuleCOSIClassifier was"
                f" fitted with {self.X_.shape[1]}."
            )
        return self.simplified_ruleset_.predict_proba(X)

    def _validate_and_create_base_ensemble(self):
        """Validate the parameter of base ensemble and if it is None,
        it set the default ensemble GradientBoostingClassifier.

        """

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than zero, "
                "got {0}.".format(self.n_estimators)
            )
        if self.base_ensemble is None:
            if is_classifier(self):
                self.base_ensemble_ = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state,
                )
            elif is_regressor(self):
                self.base_ensemble_ = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(
                    "You should choose an original classifier/regressor "
                    "ensemble to use RuleCOSI method."
                )
        self.base_ensemble_.n_estimators = self.n_estimators

        self.base_ensemble_.max_depth = self.tree_max_depth
        return clone(self.base_ensemble_)

    def class_maker(self, str):
        if str == "combine":
            return Combine(
                X_=self.X_,
                y_=self.y_,
                classes_=self.classes_,
                global_condition_map=self._global_condition_map,
                rule_heuristics=self._rule_heuristics,
            )

        elif str == "pruning":
            return SCPruning(
                X_=self.X_,
                y_=self.y_,
                cov_threshold=self.cov_threshold,
                conf_threshold=self.conf_threshold,
                classes_=self.classes_,
                global_condition_map=self._global_condition_map,
                rule_heuristics=self._rule_heuristics,
            )

        elif str == "generalize":
            return Generalize(
                X_=self.X_,
                y_=self.y_,
                cov_threshold=self.cov_threshold,
                conf_threshold=self.conf_threshold,
                confidence_level=0.25,
                classes_=self.classes_,
                global_condition_map=self._global_condition_map,
                rule_heuristics=self._rule_heuristics,
            )

        else:
            raise ValueError(
                f"Invalid argument '{str}' passed to class_maker. "
                "Expected one of: 'combine', 'pruning', 'generalize'."
            )

    def add_default_ruleset(self):
        remaining_data = self.X_.copy()
        remaining_labels = self.y_.copy()

        # 全体でカバーされているマスクを追跡する
        overall_covered_mask = np.zeros(len(remaining_data), dtype=bool)

        # 各ルールを適用
        for i, rule in enumerate(self.simplified_ruleset_.rules):
            _, covered_mask = rule.predict(remaining_data)  # 各ルールのマスク

            # カバーされたデータを更新
            overall_covered_mask |= covered_mask

        # 全てのルールでカバーされなかったデータを抽出
        uncovered_data = remaining_data[~overall_covered_mask]
        uncovered_labels = remaining_labels[~overall_covered_mask]

        # print(f"Data not covered by any rule:\n{uncovered_data}")
        # print(f"Labels not covered by any rule:\n{uncovered_labels}")

        if len(uncovered_labels) > 0:
            unique_labels, label_counts = np.unique(
                uncovered_labels, return_counts=True
            )
            majority_label = np.array([unique_labels[np.argmax(label_counts)]])
            y_class_index = np.where(self.classes_ == majority_label)[0][0]
            # print(f"Majority label: {majority_label} (Counts: {label_counts})")
            default_rule = Rule(
                conditions=[],
                class_dist=None,
                ens_class_dist=None,
                local_class_dist=None,
                logit_score=0,
                y=majority_label,
                y_class_index=y_class_index,
                classes=self.classes_,
            )
            self.simplified_ruleset_.rules.append(default_rule)
            self._rule_heuristics.compute_rule_heuristics(
                ruleset=self.simplified_ruleset_,
                recompute=True,
            )
        else:
            majority_label = None
            print("No uncovered labels to take a majority vote.")
