# This file is derived from "_rulecosi.py" from the repository:
# GitHub: https://github.com/jobregon1212/rulecosi/blob/master/rulecosi/_rulecosi.py
# Original Author: jobregon1212
# Description: Portions of code have been copied to implement rulecosi handling.
# modified: only copied the rule extaction part.

# Note: The original code is available at https://github.com/jobregon1212/rulecosi

"""RuleCOSI rule extractor.

This module contains the RuleCOSI extractor for classification problems.

The module structure is the following:

- The `BaseRuleCOSI` base class implements a common ``fit`` method
  for all the estimators in the module. This is done because in the future,
  the algorithm will work with regression problems as well.

- :class:`rulecosi.RuleCOSIClassifier` implements rule extraction from a
   variety of ensembles for classification problems.

"""

# Authors: Josue Obregon <jobregon@khu.ac.kr>
#
#
# License: TBD

import copy

from .rule_extraction import RuleExtractorFactory
from .rule_heuristics import RuleHeuristics


class RuleExtraction():

    def __init__(self,
                 base_ensemble_=None,
                 column_names=None,
                 classes_=None,
                 X_=None,
                 y_=None,
                 cov_threshold=0.0,
                 conf_threshold=0.5,
                 float_threshold=1e-6,
                 verbose=0
                 ):

        self.base_ensemble_=base_ensemble_
        self.column_names=column_names
        self.classes_=classes_
        self.X_=X_
        self.y_=y_
        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.float_threshold=float_threshold
        self.verbose=verbose
    
    def rule_extraction(self):
        self._bad_combinations=None
        self._good_combinations=None
        self._rule_extractor = None
        self._rule_heuristics = None
        self._global_condition_map = None
        self.original_rulesets_ = None

        # First step is extract the rules
        self._rule_extractor = RuleExtractorFactory.get_rule_extractor(
            self.base_ensemble_, self.column_names,
            self.classes_, self.X_, self.y_, self.float_threshold)
        if self.verbose > 0:
            print(
                f'Extracting rules from {type(self.base_ensemble_).__name__} '
                f'base ensemble...')
        self.original_rulesets_, \
        self._global_condition_map = self._rule_extractor.extract_rules()
        processed_rulesets = copy.deepcopy(self.original_rulesets_)

        # ルールの可視化
        # print(processed_rulesets)

        # We create the heuristics object which will compute all the
        # heuristics related measures
        self._rule_heuristics = RuleHeuristics(X=self.X_, y=self.y_,
                                               classes_=self.classes_,
                                               condition_map=
                                               self._global_condition_map,
                                               cov_threshold=self.cov_threshold,
                                               conf_threshold=
                                               self.conf_threshold)
        if self.verbose > 0:
            print(f'Initializing sets and computing condition map...')
        self._initialize_sets()

        if str(
                self.base_ensemble_.__class__) == \
                "<class 'catboost.core.CatBoostClassifier'>":
            for ruleset in processed_rulesets:
                for rule in ruleset:
                    new_A = self._remove_opposite_conditions(set(rule.A),
                                                             rule.class_index)
                    rule.A = new_A

        for ruleset in processed_rulesets:
            self._rule_heuristics.compute_rule_heuristics(
                ruleset, recompute=True)
            
        return processed_rulesets, self._global_condition_map

    def _initialize_sets(self):
        """ Initialize the sets that are going to be used during the
        combination and simplification process This includes the set of good
        combinations G and bad combinations B. It also includes the bitsets
        for the training data as well as the bitsets for each of the conditions
        """
        self._bad_combinations = set()
        self._good_combinations = dict()
        self._rule_heuristics.initialize_sets()
