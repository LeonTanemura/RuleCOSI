from rule_making import RuleSet, Rule


def generalize_ruleset(
    ruleset, beta, alpha, rule_heuristics, global_condition_map, classes_
):
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
        if rule_heuristics._cond_cov_dict is None:
            print("Initializing condition coverage dictionary...")
            rule_heuristics.initialize_sets()

        heuristics_dict, _ = rule_heuristics.get_conditions_heuristics(rule.A)

        baseline_error = 1 - max(heuristics_dict["conf"])
        min_error = 0

        # 一般化ループ
        while min_error <= baseline_error and rule.A:
            # 各条件を削除して誤差を評価
            errors = []
            for cond in rule.A:
                modified_rule = remove_condition(rule, cond)
                new_heuristics, _ = rule_heuristics.get_conditions_heuristics(
                    modified_rule.A
                )
                new_error = 1 - max(new_heuristics["conf"])
                errors.append((cond, new_error))

            # 最小の誤差を持つ条件を選択
            best_condition, min_error = min(errors, key=lambda x: x[1])

            if min_error <= baseline_error:
                rule.A = remove_condition(rule, best_condition).A  # 条件を削除
                baseline_error = min_error
                min_error = 0

        # カバレッジと信頼度が閾値を満たしているか確認
        coverage = heuristics_dict["cov"]
        confidence = max(heuristics_dict["conf"])
        if coverage > beta and confidence > alpha:
            generalized_rules.append(rule)

    generalized_ruleset = RuleSet(
        rules=list(generalized_rules),
        condition_map=global_condition_map,
        classes=classes_,
    )
    generalized_ruleset = compute_heuristics(generalized_ruleset, rule_heuristics)
    # 信頼度とカバレッジでソート
    generalized_ruleset.rules.sort(
        key=lambda rule: (rule.cov, rule.conf, rule.str, id(rule)),
        reverse=True,
    )

    return generalized_ruleset


def remove_condition(rule, condition):
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

    # ヒューリスティクスを再計算


def compute_heuristics(ruleset, rule_heuristics):
    if not rule_heuristics:
        raise ValueError("Rule heuristics object not initialized.")

    rule_heuristics.compute_rule_heuristics(
        ruleset=ruleset,
        not_cov_mask=rule_heuristics.ones,
        recompute=True,
    )
    return ruleset
