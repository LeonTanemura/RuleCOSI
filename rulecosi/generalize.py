from rule_making import RuleSet, Rule

from math import sqrt
from scipy.stats import norm


class Generalize:

    def __init__(
        self,
        cov_threshold=0.0,
        conf_threshold=0.5,
        confidence_level=0.25,
        X_=None,
        y_=None,
        classes_=None,
        global_condition_map=None,
        rule_heuristics=None,
    ):
        self.X_ = X_
        self.y_ = y_
        self.classes_ = classes_
        self.global_condition_map = global_condition_map
        self.rule_heuristics = rule_heuristics
        self.not_cov_mask = self.rule_heuristics.ones
        self.alpha = conf_threshold
        self.beta = cov_threshold
        self.c_level = confidence_level

    def generalize_ruleset(self, ruleset):
        """
        論文に基づいてルールセットを一般化する。

        :param ruleset: 初期ルールセット（リスト形式）
        :param beta: カバレッジの閾値
        :param alpha: 信頼度の閾値
        :param dataset: データセット（リスト形式）
        :param rule_heuristics: ルールのカバレッジと信頼度を計算するためのオブジェクト
        :return: 一般化されたルールセット
        """
        generalized_rules = []

        for rule in ruleset:
            if self.rule_heuristics._cond_cov_dict is None:
                print("Initializing condition coverage dictionary...")
                self.rule_heuristics.initialize_sets()

            heuristics_dict, _ = self.rule_heuristics.get_conditions_heuristics(rule.A)

            baseline_error = self.Error(rule, self.c_level)
            min_error = 0

            # 一般化ループ
            while min_error <= baseline_error and rule.A:
                # 各条件を削除して誤差を評価
                errors = []
                for cond in rule.A:
                    modified_rule = self.remove_condition(rule, cond)
                    new_heristics, _ = self.rule_heuristics.get_conditions_heuristics(
                        modified_rule.A
                    )
                    modified_rule.set_heuristics(new_heristics)
                    new_error = self.Error(modified_rule, self.c_level)
                    errors.append((cond, new_error))

                # 最小の誤差を持つ条件を選択
                best_condition, min_error = min(errors, key=lambda x: x[1])

                if min_error <= baseline_error:
                    rule.A = self.remove_condition(rule, best_condition).A  # 条件を削除
                    baseline_error = min_error
                    min_error = 0

            # カバレッジと信頼度が閾値を満たしているか確認
            coverage = heuristics_dict["cov"]
            confidence = max(heuristics_dict["conf"])
            if coverage > self.beta and confidence > self.alpha:
                generalized_rules.append(rule)

        self.generalized_ruleset = RuleSet(
            rules=list(generalized_rules),
            condition_map=self.global_condition_map,
            classes=self.classes_,
        )
        self.compute_heuristics(self.generalized_ruleset)
        # 信頼度とカバレッジでソート
        self.generalized_ruleset.rules.sort(
            key=lambda rule: (rule.cov, rule.conf, rule.str, id(rule)),
            reverse=True,
        )

        return self.generalized_ruleset

    def remove_condition(self, rule, condition):
        """Remove a condition from the rule and return a new Rule object."""
        # If A is a frozenset, convert it to a set
        if isinstance(rule.A, frozenset):
            new_conditions = set(rule.A)  # Convert frozenset to set
        else:
            new_conditions = rule.A.copy()  # If it's already a set, make a copy

        new_conditions.discard(
            condition
        )  # Remove the condition (discard does not raise an error)

        # Re-create the Rule with the updated conditions
        return Rule(
            new_conditions,
            rule.class_dist,
            rule.ens_class_dist,
            rule.local_class_dist,
            rule.logit_score,
            rule.y,
            rule.class_index,
            classes=rule.classes,  # Ensure classes are passed correctly
        )

    def Error(self, r, c):
        """
        Calculate the upper bound of the generalization error for a rule.

        Parameters:
        r: Rule object with attributes A (conditions) and conf (confidence).
        c: Confidence level (e.g., 0.25 for 25%).

        Returns:
        Upper bound of the generalization error.
        """
        nr = r.n_samples  # Total number of training records covered by rule r
        ntr = nr[0] + nr[1]
        er = 1 - r.conf  # Training error, where conf is the rule's confidence
        p = 1 - c
        z_c_2 = norm.ppf(1 - p / 2)  # Standardized value from the normal distribution

        # Calculate the upper bound of the error
        e_upper = (
            er
            + (z_c_2**2) / (2 * ntr)
            + z_c_2 * sqrt((er * (1 - er)) / ntr + (z_c_2**2) / (4 * ntr**2))
        ) / (1 + (z_c_2**2) / ntr)
        return e_upper

    # ヒューリスティクスを再計算
    def compute_heuristics(self, ruleset):
        if not self.rule_heuristics:
            raise ValueError("Rule heuristics object not initialized.")

        self.rule_heuristics.compute_rule_heuristics(
            ruleset=ruleset,
            not_cov_mask=self.rule_heuristics.ones,
            recompute=True,
        )
