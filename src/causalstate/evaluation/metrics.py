from causalstate.world.gridworld import ALL_VARS, CAUSAL_ORACLE


def recovery_score(selected, oracle=CAUSAL_ORACLE) -> dict:
    selected_set = set(selected)
    oracle_set = set(oracle)
    all_vars_set = set(ALL_VARS)

    unknown_selected = selected_set - all_vars_set
    unknown_oracle = oracle_set - all_vars_set

    if unknown_selected or unknown_oracle:
        raise ValueError(
            f"Unknown variables: {sorted(unknown_selected | unknown_oracle)}"
        )

    tp = selected_set & oracle_set
    fp = selected_set - oracle_set
    fn = oracle_set - selected_set

    tp_count = len(tp)
    fp_count = len(fp)
    fn_count = len(fn)

    precision = (
        tp_count / (tp_count + fp_count)
        if (tp_count + fp_count) > 0
        else 0.0
    )
    recall = (
        tp_count / (tp_count + fn_count)
        if (tp_count + fn_count) > 0
        else 0.0
    )
    f1 = (
        (2 * tp_count) / (2 * tp_count + fp_count + fn_count)
        if (2 * tp_count + fp_count + fn_count) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
    }
