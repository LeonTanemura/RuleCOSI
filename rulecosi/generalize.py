from rule_making import RuleSet, Rule


def generalize_ruleset(ruleset, alpha, beta, confidence_level, rule_heuristics):
    """Generalizes a given ruleset according to specified thresholds and heuristics."""
    for rule in ruleset:
        if not rule.A:  # If the rule has no conditions, skip it
            continue

        if rule_heuristics._cond_cov_dict is None:
            print("Initializing condition coverage dictionary...")
            rule_heuristics.initialize_sets()

        heuristics_dict, _ = rule_heuristics.get_conditions_heuristics(rule.A)

        baseline_error = 1 - max(heuristics_dict["conf"])
        min_error = baseline_error

        while min_error <= baseline_error and rule.A:
            errors = [
                (
                    cond,
                    1
                    - max(
                        rule_heuristics.get_conditions_heuristics(
                            remove_condition(rule, cond).A
                        )[0]["conf"]
                    ),
                )
                for cond in rule.A
            ]
            best_condition, min_error = min(errors, key=lambda x: x[1])

            if min_error <= baseline_error:
                rule.A = remove_condition(
                    rule, best_condition
                ).A  # Update rule.A after removing the condition
                baseline_error = min_error

        # Check if rule meets the coverage and confidence thresholds
        if not (heuristics_dict["cov"] > beta and max(heuristics_dict["conf"]) > alpha):
            ruleset.remove(rule)

    # Sort the rules based on confidence and coverage
    # ruleset.rules.sort(
    #     key=lambda r: (
    #         max(rule_heuristics.get_conditions_heuristics(r.A)[0]["conf"]),
    #         rule_heuristics.get_conditions_heuristics(r.A)[0]["cov"],
    #     ),
    #     reverse=True,
    # )
    return ruleset


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
