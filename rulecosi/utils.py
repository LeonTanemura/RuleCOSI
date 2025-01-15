from operator import attrgetter


def sort_ruleset(ruleset):
    """Sort the ruleset in place according to the rule_order parameter.

    :param ruleset: ruleset to be ordered
    """
    sorting_list = ["conf", "cov", "supp"]
    if len(ruleset.rules) == 0:
        return

    ruleset.rules.sort(key=lambda rule: (-1 * len(rule.A), rule.str))
    for attr in sorting_list:
        ruleset.rules.sort(key=attrgetter(attr), reverse=True)
