import numpy as np


def _simplify_conditions(conditions, _global_condition_map):

    cond_map = _global_condition_map  # For readability
    # Create a list in the format [(att_index, 'OPERATOR'), 'cond_id']
    att_op_list = [
        [(cond[1].att_index, cond[1].op.__name__), cond[0]] for cond in conditions
    ]
    att_op_list = np.array(att_op_list, dtype=object)

    dict_red_cond = {
        i[0]: [
            att_op_list[idx][1]
            for idx in range(len(att_op_list))
            if att_op_list[idx][0] == i[0]
        ]
        for i in att_op_list
    }

    # Iterate over attributes and operators with multiple conditions
    gen_red_cond = (
        (att_op, conds) for (att_op, conds) in dict_red_cond.items() if len(conds) > 1
    )

    for att_op, conds in gen_red_cond:
        tup_att_op = att_op
        list_conds = {cond_map[int(id_)] for id_ in conds}

        # Retain the most restrictive condition
        if tup_att_op[1] in ["lt", "le"]:
            edge_condition = max(list_conds, key=lambda item: item.value)
        elif tup_att_op[1] in ["gt", "ge"]:
            edge_condition = min(list_conds, key=lambda item: item.value)

        # Remove all other conditions
        list_conds.remove(edge_condition)
        [conditions.remove((hash(cond), cond)) for cond in list_conds]

    return frozenset(conditions)


def _simplify_rulesets(tree, _global_condition_map):

    for ruleset in tree:
        for rule in ruleset:
            rule.A = _simplify_conditions(set(rule.A), _global_condition_map)

    return tree
