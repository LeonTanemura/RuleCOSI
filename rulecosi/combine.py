import numpy as np
from rule_making import Rule
from ._simplify_rulesets import _simplify_conditions


class Combine:

    def __init__(self, cov_threshold=0.0, conf_threshold=0.5, X_=None, y_=None, classes_=None): 
        self.X_ = X_
        self.y_ = y_
        self.classes_ = classes_
        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold

    def combine_rulesets(self, simplified_ruleset_, global_condition_map, rule_heuristics):
        self.simplified_ruleset_ = simplified_ruleset_
        self.global_condition_map = global_condition_map
        self.rule_heuristics = rule_heuristics
        self.combined_rulesets = []

        # すべてのルールセットを結合
        for idx, ruleset in enumerate(self.simplified_ruleset_):
            rs1 = ruleset
            for rs2 in self.simplified_ruleset_[idx+1:]:
                combined_result = self.combine(rs1, rs2)
                self.combined_rulesets.append(combined_result)
                # self.combined_rulesets.extend(combined_result)  # リストに展開して追加
                # print("combine len : ", len(self.combined_rulesets))
        print(self.combined_rulesets)
        
        return self.combined_rulesets
        
    def combine(self, rs1, rs2):
        combined_ruleset = set()
        # print("rs1 : ", len(rs1.rules))
        # print("rs2 : ", len(rs2.rules))
        for r1 in rs1:
            for r2 in rs2:
                # print("r1 conditions: ", r1.A)
                # print("r2 conditions: ", r2.A)
                # 新しいルール条件を結合
                new_rule_condition = set(r1.A).union(set(r2.A))
                new_rule_condition = _simplify_conditions(new_rule_condition, self.global_condition_map)
                # print("new_rule condition: ", new_rule_condition)

                # 新しいクラス分布を計算
                if r1.class_dist is not None and r2.class_dist is not None:
                    new_rule_class_dist = (r1.class_dist + r2.class_dist) / 2
                else:
                    new_rule_class_dist = r1.class_dist or r2.class_dist

                # 新しいルールのヒューリスティクスを結合
                heuristics_dict = self.rule_heuristics.combine_heuristics(
                    r1.heuristics_dict, r2.heuristics_dict
                )

                y_class_index = np.argmax(new_rule_class_dist).item()
                y = np.array([self.classes_[y_class_index]])

                # print("Combined heuristics:", heuristics_dict)
                # print(f"New rule class: {y}, Coverage: {heuristics_dict['cov']}, Confidence: {heuristics_dict['conf'][y_class_index]}")

                if heuristics_dict['cov'] > self.cov_threshold and heuristics_dict['conf'][y_class_index] > self.conf_threshold:
                    new_rule = Rule(
                        conditions=new_rule_condition,
                        class_dist=new_rule_class_dist,
                        ens_class_dist=new_rule_class_dist,
                        local_class_dist=new_rule_class_dist,
                        logit_score=0,
                        y=y,
                        y_class_index=y_class_index,
                        classes=self.classes_
                    )
                    new_rule.set_heuristics(heuristics_dict)
                    combined_ruleset.add(new_rule)

        # print("combined: ", len(combined_ruleset))

        return combined_ruleset
