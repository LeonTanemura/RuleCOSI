# This file is derived from "rule_extraction.py" from the repository:
# GitHub: https://github.com/jobregon1212/rulecosi/blob/master/rulecosi/rule_extraction.py
# Original Author: jobregon1212
# Description: Portions of code have been copied to implement rule extraction handling.

# Note: The original code is available at https://github.com/jobregon1212/rulecosi

""" This module contains the functions used for extracting the rules for
different type of base ensembles.

The module structure is the following:

- The `BaseRuleExtractor` base class implements a common ``get_base_ruleset``
  and ``recursive_extraction``  method for all the extractors in the module.

    - :class:`rule_extraction.DecisionTreeRuleExtractor` implements rule
        extraction from a single decision tree

    - :class:`rule_extraction.ClassifierRuleExtractor` implements rule
        extraction from a classifier Ensembles such as Bagging and
        Random Forests

    - :class:`rule_extraction.GBMClassifierRuleExtractor` implements rule
        extraction from sklearn GBM classifier and works as base class for the
        other GBM implementations

        - :class:`rule_extraction.XGBClassifierExtractor` implements rule
            extraction from XGBoost classifiers

        - :class:`rule_extraction.LGBMClassifierExtractor` implements rule
            extraction from Light GBM classifiers

        - :class:`rule_extraction.CatBoostClassifierExtractor` implements rule
            extraction from CatBoost classifiers


"""

import json
import copy
import operator
from os import path
from abc import ABCMeta

import numpy as np
from math import copysign

from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier

from .rules import RuleSet, Condition, Rule


class BaseRuleExtractor(metaclass=ABCMeta):
    """Base abstract class for a rule extractor from tree ensembles"""

    def __init__(self, _ensemble, _column_names, classes_, X, y, float_threshold):
        self._column_names = _column_names
        self.classes_ = classes_
        self._ensemble = _ensemble
        self.X = X
        self.y = y
        self.float_threshold = float_threshold
        _, counts = np.unique(self.y, return_counts=True)
        self.class_ratio = counts.min() / counts.max()

    def get_tree_dict(self, base_tree, n_nodes=0):
        """Create a dictionary with the information inside the base_tree

        :param base_tree: :class: `sklearn.tree.Tree` object which is an array
            representation of a tree

        :param n_nodes: number of nodes in the tree

        :return: a dictionary containing the information of the base_tree
        """
        return {
            "children_left": base_tree.tree_.children_left,
            "children_right": base_tree.tree_.children_right,
            "feature": base_tree.tree_.feature,
            "threshold": base_tree.tree_.threshold,
            "value": base_tree.tree_.value,
            "n_samples": base_tree.tree_.weighted_n_node_samples,
            "n_nodes": base_tree.tree_.node_count,
        }

    def get_base_ruleset(
        self, tree_dict, class_index=None, condition_map=None, tree_index=None
    ):
        """

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        :param class_index: Right now is not used but it will be used
        when multiclass is supported

        :param condition_map: dictionary of <condition_id, Condition>,
        default=None. Dictionary of Conditions extracted from all the
        ensembles.condition_id is an integer uniquely identifying the Condition.

        :param tree_index: index of the tree in the ensemble

        :return:   a :class:`rulecosi.rules.RuleSet' object
        """

        if condition_map is None:
            condition_map = dict()  # dictionary of conditions A

        extracted_rules = self.recursive_extraction(
            tree_dict,
            tree_index,
            class_index=class_index,
            node_index=0,
            condition_map=condition_map,
            condition_set=set(),
        )
        return RuleSet(extracted_rules, condition_map, classes=self.classes_)

    def recursive_extraction(
        self,
        tree_dict,
        tree_index=0,
        class_index=None,
        node_index=0,
        condition_map=None,
        condition_set=None,
    ):
        """Recursive function for extracting a ruleset from a tree

        :param class_index: class index for which this tree predicts to
        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        :param tree_index: index of the tree in the ensemble

        :param node_index: the index of the leaf node

        :param condition_map: condition_map: dictionary of <condition_id,
        Condition>, default=None Dictionary of Conditions extracted from all
        the ensembles. condition_id is an integer uniquely identifying the
        Condition.

        :param condition_set:  set of :class:`rulecosi.rule.Condition` objects

        :return: array of :class:`rulecosi.rules.Rule` objects
        """
        if condition_map is None:
            condition_map = dict()
        if condition_set is None:
            condition_set = set()
        rules = []
        children_left = tree_dict["children_left"]
        children_right = tree_dict["children_right"]
        feature = tree_dict["feature"]
        threshold = tree_dict["threshold"]

        # leaf node so a rule is created
        if children_left[node_index] == children_right[node_index]:
            weights = None
            logit_score = None
            new_rule = self.create_new_rule(
                node_index,
                tree_dict,
                condition_set,
                logit_score,
                weights,
                tree_index,
                class_index=class_index,
            )
            rules.append(new_rule)
        else:
            # create condition, add it to the condition_set and get conditions from left and right child
            att_name = None
            if self._column_names is not None:
                att_name = self._column_names[feature[node_index]]
            condition_set_left = copy.copy(condition_set)
            # condition_set_left = copy.copy(condition_set)
            # determine operators
            op_left, op_right = self.get_split_operators()

            # -0 problem solution
            split_value = threshold[node_index]
            if abs(split_value) < self.float_threshold:
                split_value = copysign(self.float_threshold, split_value)
                # split_value=0
                # print(split_value)
            new_condition_left = Condition(
                feature[node_index],
                op_left,
                # threshold[node_index],
                split_value,
                att_name,
            )
            condition_map[hash(new_condition_left)] = new_condition_left
            # condition_set_left.add(hash(new_condition_left))
            condition_set_left.add((hash(new_condition_left), new_condition_left))
            left_rules = self.recursive_extraction(
                tree_dict,
                tree_index,
                node_index=children_left[node_index],
                class_index=class_index,
                condition_set=condition_set_left,
                condition_map=condition_map,
            )
            rules = rules + left_rules

            condition_set_right = copy.copy(condition_set)
            # condition_set_right = copy.copy(condition_set)
            new_condition_right = Condition(
                feature[node_index],
                op_right,
                # threshold[node_index],
                split_value,
                att_name,
            )
            condition_map[hash(new_condition_right)] = new_condition_right
            # condition_set_right.add(hash(new_condition_right))
            condition_set_right.add((hash(new_condition_right), new_condition_right))
            right_rules = self.recursive_extraction(
                tree_dict,
                tree_index,
                class_index=class_index,
                node_index=children_right[node_index],
                condition_set=condition_set_right,
                condition_map=condition_map,
            )
            rules = rules + right_rules
        return rules

    def get_split_operators(self):
        """Return the operator applied for the left and right branches of
        the tree. This function is needed because different implementations
        of trees use different operators for the children nodes.

        :return: a tuple containing the left and right operator used for
        creating conditions
        """
        op_left = operator.le  # Operator.LESS_OR_EQUAL_THAN
        op_right = operator.gt  # Operator.GREATER_THAN
        return op_left, op_right


def get_class_dist(raw_to_proba):
    return np.array([1 - raw_to_proba.item(), raw_to_proba.item()])


class GBMClassifierRuleExtractor(BaseRuleExtractor):
    """Rule extraction for a Gradient Boosting Tree ensemble classifier.
    This class accept just sklearn GBM implementation.

    Parameters
    ----------
    base_ensemble: BaseEnsemble object, default = None
        A BaseEnsemble estimator object. The supported types are:
        - :class:`sklearn.ensemble.GradientBoostingClassifier`


    column_names: array of string, default=None Array of strings with the
    name of the columns in the data. This is useful for displaying the name
    of the features in the generated rules.

    classes: ndarray, shape (n_classes,)
        The classes seen when fitting the ensemble.

    X: array-like, shape (n_samples, n_features)
        The training input samples.

    """

    def extract_rules(self):
        """Main method for extracting the rules of tree ensembles

        :return: an array of :class:`rulecosi.rules.RuleSet'
        """
        rulesets = []
        global_condition_map = dict()
        for tree_index, base_trees in enumerate(self._ensemble):
            original_ruleset = None
            for class_index, base_tree in enumerate(base_trees):
                if len(self.classes_) == 2:  # binary classification
                    original_ruleset = self.get_base_ruleset(
                        self.get_tree_dict(base_tree),
                        class_index=None,
                        tree_index=tree_index,
                    )
                else:  # multi-class classification
                    print(
                        f"Error: Currently, only binary classification is supported. Number of classes: {len(self.classes_)}"
                    )
                    raise ValueError(
                        f"Unsupported number of classes. Currently, only binary classification is supported. Number of classes: {len(self.classes_)}"
                    )

            rulesets.append(original_ruleset)
            global_condition_map.update(original_ruleset.condition_map)
        return rulesets, global_condition_map

    def create_new_rule(
        self,
        node_index,
        tree_dict,
        condition_set=None,
        logit_score=None,
        weights=None,
        tree_index=None,
        class_index=None,
    ):
        """Creates a new rule with all the information in the parameters

        :param node_index: the index of the leaf node

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        :param condition_set: set of :class:`rulecosi.rule.Condition` objects
        of the new rule

        :param logit_score: logit_score of the rule (only applies for
        Gradient Boosting Trees)

        :param weights: weight of the new rule

        :param tree_index: index of the tree inside the ensemble

        :param class_index: index of the class that this rule predicts

        :return: a :class:`rulecosi.rules.Rule` object
        """
        if condition_set is None:
            condition_set = {}
        value = tree_dict["value"]
        n_samples = tree_dict["n_samples"]

        if tree_index == 0:
            init = self._get_gbm_init()
        else:
            init = np.zeros(value[node_index].shape)

        logit_score = init + value[node_index]

        if len(self.classes_) == 2:
            raw_to_proba = expit(logit_score)
            class_dist = get_class_dist(raw_to_proba)
        else:
            # Print error message for unsupported number of classes
            print(
                f"Error: Currently, only binary classification is supported. Number of classes: {len(self.classes_)}"
            )

            # Raise an exception
            raise ValueError(
                f"Unsupported number of classes. Currently, only binary classification is supported. Number of classes: {len(self.classes_)}"
            )

        # predict y_class_index = np.argmax(class_dist).item()
        # y_class_index = np.argmax(class_dist).item()
        y_class_index = np.argmax(class_dist)
        y = np.array([self.classes_[y_class_index]])
        return Rule(
            frozenset(condition_set),
            class_dist=class_dist,
            ens_class_dist=class_dist,
            logit_score=logit_score,
            y=y,
            y_class_index=y_class_index,
            n_samples=n_samples[node_index],
            classes=self.classes_,
            weight=weights,
        )

    def _get_gbm_init(self):
        """get the initial estimate of a GBM ensemble

        :return: a double value of the initial estimate of the GBM ensemble
        """
        return self._ensemble._raw_predict_init(self.X[0].reshape(1, -1))


class XGBClassifierExtractor(GBMClassifierRuleExtractor):
    """Rule extraction for a Gradient Boosting Tree ensemble classifier.
    This class accept only XGB implementation

    Parameters
    ----------
    base_ensemble: BaseEnsemble object, default = None
        A BaseEnsemble estimator object. The supported types are:
            - :class:`xgboost.XGBClassifier`

    column_names: array of string, default=None Array of strings with the
    name of the columns in the data. This is useful for displaying the name
    of the features in the generated rules.

    classes: ndarray, shape (n_classes,)
        The classes seen when fitting the ensemble.

    X: array-like, shape (n_samples, n_features)
        The training input samples.

    """

    def extract_rules(self):
        """Main method for extracting the rules of tree ensembles

        :return: an array of :class:`rulecosi.rules.RuleSet'
        """
        rulesets = []
        global_condition_map = dict()
        booster = self._ensemble.get_booster()
        booster.feature_names = self._column_names.to_list()
        xgb_tree_dicts = booster.get_dump(dump_format="json")
        n_nodes = (
            booster.trees_to_dataframe()[["Tree", "Node"]]
            .groupby("Tree")
            .count()
            .to_numpy()
        )

        if len(self.classes_) == 2:  # binary classification
            for t_idx, xgb_t_dict in enumerate(xgb_tree_dicts):
                original_ruleset = self.get_base_ruleset(
                    self.get_tree_dict(xgb_t_dict, n_nodes[t_idx]),
                    class_index=0,
                    tree_index=t_idx,
                )
                rulesets.append(original_ruleset)
                global_condition_map.update(original_ruleset.condition_map)
            return rulesets, global_condition_map
        else:
            # Print error message for unsupported number of classes
            print(
                f"Error: Currently, only binary classification is supported. Number of classes: {len(self.classes_)}"
            )

            # Raise an exception
            raise ValueError(
                f"Unsupported number of classes. Currently, only binary classification is supported. Number of classes: {len(self.classes_)}"
            )

    # def _get_class_dist(self, raw_to_proba):
    #     return np.array([raw_to_proba.item(), 1 - raw_to_proba.item()])

    def get_tree_dict(self, base_tree, n_nodes=0):
        """Create a dictionary with the information inside the base_tree

        :param base_tree: :class: `sklearn.tree.Tree` object wich is an array
        representation of a tree

        :param n_nodes: number of nodes in the tree

        :return: a dictionary conatining the information of the base_tree
        """
        tree_dict = {
            "children_left": np.full(n_nodes, fill_value=-1),
            "children_right": np.full(n_nodes, fill_value=-1),
            "feature": np.full(n_nodes, fill_value=0),
            "threshold": np.full(n_nodes, fill_value=0.0),
            "value": np.full(n_nodes, fill_value=0.0),
            "n_samples": np.full(n_nodes, fill_value=-1),
            "n_nodes": n_nodes,
        }

        tree = json.loads(base_tree)
        self._populate_tree_dict(tree, tree_dict)
        return tree_dict

    def _populate_tree_dict(self, tree, tree_dict):
        """Populate the tree dictionary specifically for this type of GBM
        implementation. This is needed because each GBM implementation output
        the trees in different formats

        :param tree: the current tree to be used as a source

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        """
        node_id = tree["nodeid"]
        if "leaf" in tree:
            tree_dict["value"][node_id] = tree["leaf"]
            return
        if "children" in tree:
            tree_dict["children_left"][node_id] = tree["children"][0]["nodeid"]
            tree_dict["children_right"][node_id] = tree["children"][1]["nodeid"]
            # tree_dict['feature'][node_id] = int(tree['split'][1:])
            # if not str.isdigit(tree['split']):
            # new change 2023/03/14 . setting feature names before
            # dumping allows to ensure feature names are preserved from
            # self.column_names
            tree_dict["feature"][node_id] = np.where(
                self._column_names == tree["split"]
            )[
                0
            ].item()  # 2021/23/06 change, the split directly

            # tree_dict['feature'][node_id] = \
            #     int(tree['split'].replace('f', ''))
            # 2022/04/16 change obtain directly the feature index
            # np.where(self._column_names == tree['split'])[
            #     0].item()  # 2021/23/06 change, the split directly

            # print('feature: ', tree['split'])
            # print('node_id: ', tree_dict['feature'][node_id])
            # else:
            #     tree_dict['feature'][node_id] = int(
            #         tree['split'])  # 2021/23/06 change, the split directly
            tree_dict["threshold"][node_id] = tree["split_condition"]
            self._populate_tree_dict(tree["children"][0], tree_dict)
            self._populate_tree_dict(tree["children"][1], tree_dict)

    def get_split_operators(self):
        """Return the operator applied for the left and right branches of
        the tree. This function is needed because different implementations
        of trees use different operators for the children nodes.

        :return: a tuple containing the left and right operator used for
        creating conditions
        """
        op_left = operator.lt  # Operator.LESS_THAN
        op_right = operator.ge  # Operator.GREATER_OR_EQUAL_THAN
        return op_left, op_right

    def _get_gbm_init(self):
        """get the initial estimate of a GBM ensemble

        :return: a double value of the initial estimate of the GBM ensemble
        """
        if self._ensemble.base_score is None:
            return self.class_ratio
        else:
            return self._ensemble.base_score


class RuleExtractorFactory:
    """Factory class for getting an implementation of a BaseRuleExtractor"""

    def get_rule_extractor(base_ensemble, column_names, classes, X, y, float_threshold):
        """

        :param base_ensemble: BaseEnsemble object, default = None
            A BaseEnsemble estimator object. The supported types are:
           - :class:`sklearn.ensemble.RandomForestClassifier`
           - :class:`sklearn.ensemble.BaggingClassifier`
           - :class:`sklearn.ensemble.GradientBoostingClassifier`
           - :class:`xgboost.XGBClassifier`
           - :class:`catboost.CatBoostClassifier`
           - :class:`lightgbm.LGBMClassifier`

        :param column_names: array of string, default=None Array of strings
        with the name of the columns in the data. This is useful for
        displaying the name of the features in the generated rules.

        :param classes: ndarray, shape (n_classes,)
            The classes seen when fitting the ensemble.

        :param X: array-like, shape (n_samples, n_features)
         The training input samples.

        :return: A BaseRuleExtractor class implementation instantiated object
        to be used for extracting rules from trees
        """
        if isinstance(base_ensemble, GradientBoostingClassifier):
            return GBMClassifierRuleExtractor(
                base_ensemble, column_names, classes, X, y, float_threshold
            )
        elif str(base_ensemble.__class__) == "<class 'xgboost.sklearn.XGBClassifier'>":
            return XGBClassifierExtractor(
                base_ensemble, column_names, classes, X, y, float_threshold
            )
        else:
            # Print the error message
            print(f"Error: Unsupported base_ensemble type: {base_ensemble.__class__}")

            # Optionally, raise an exception
            raise ValueError(
                f"Unsupported base_ensemble type: {base_ensemble.__class__}"
            )
