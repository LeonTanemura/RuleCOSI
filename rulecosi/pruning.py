import numpy as np
from rule_making import RuleSet


class SCPruning:

    def __init__(
        self,
        cov_threshold=0.0,
        conf_threshold=0.5,
        X_=None,
        y_=None,
        classes_=None,
        global_condition_map=None,
        rule_heuristics=None,
    ):
        self.X_ = X_
        self.y_ = y_
        self.classes_ = classes_
        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.global_condition_map = global_condition_map
        self.rule_heuristics = rule_heuristics

        self.remaining_data = X_.copy() if X_ is not None else None
        self.remaining_labels = y_.copy() if y_ is not None else None
        self.not_cov_mask = self.rule_heuristics.ones if self.rule_heuristics else None

    def sequential_covering_pruning(self, unpruned_rulesets):

        self.unpruned_rulesets = unpruned_rulesets
        self.pruned_ruleset = set()

        while len(self.unpruned_rulesets.rules) > 0 and len(self.remaining_data) > 0:
            self.compute_heuristics(self.unpruned_rulesets)
            self.sort_heuristics(self.unpruned_rulesets)

            # ソート後のヒューリスティクスを表示
            # for i, rule in enumerate(self.unpruned_rulesets.rules[:5]):
            #     print(f"順位: {i+1}")
            #     print(f"  カバレッジ (Coverage): {rule.cov}")
            #     print(f"  信頼度 (Confidence): {rule.conf}")
            #     print(f"  サポート率 (Support): {rule.supp}")
            #     print(f"  ルール条件セット (Rule Conditions): {rule}")
            #     print("-" * 10)

            found_rule = False
            for i, rule in enumerate(self.unpruned_rulesets.rules):
                result, self.not_cov_mask = self.rule_heuristics.rule_is_accurate(
                    rule, self.not_cov_mask
                )
                if result:
                    # print(f"Index of pruned rules : {i}")
                    self.pruned_ruleset.add(rule)
                    # print(
                    #     f"unpruned ruleset length: {len(self.unpruned_rulesets.rules)}"
                    # )
                    # print(f"dataset length: {len(self.remaining_data)}")
                    self.unpruned_rulesets.rules.remove(rule)
                    _, covered_mask = rule.predict(self.remaining_data)
                    self.remaining_data = self.remaining_data[~covered_mask]
                    self.remaining_labels = self.remaining_labels[~covered_mask]
                    found_rule = True
                    break

            if not found_rule:
                break

        self.pruned_ruleset = RuleSet(
            rules=list(self.pruned_ruleset),
            condition_map=self.global_condition_map,
            classes=self.classes_,
        )

        self.compute_heuristics(self.pruned_ruleset)
        self.sort_heuristics(self.pruned_ruleset)

        return self.pruned_ruleset

    # ヒューリスティクスを再計算
    def compute_heuristics(self, ruleset):
        if not self.rule_heuristics:
            raise ValueError("Rule heuristics object not initialized.")

        self.rule_heuristics.compute_rule_heuristics(
            ruleset=ruleset,
            not_cov_mask=self.not_cov_mask,
            recompute=True,
        )

    def sort_heuristics(self, ruleset):
        ruleset.rules.sort(
            key=lambda rule: (rule.cov, rule.conf, rule.str, id(rule)),
            reverse=True,
        )
