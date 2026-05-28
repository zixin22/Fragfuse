def check_access(list_of_database:dict, summary_of_database:dict):
    access_denied = False
    inaccessible = {}

    for key, value in summary_of_database.items():
        if key not in list_of_database:
            access_denied = True
            inaccessible[key] = value
        else:
            inaccessible_column_temp = list(set(value) - set(list_of_database[key]))
            if len(inaccessible_column_temp) > 0:
                access_denied = True
                inaccessible[key] = inaccessible_column_temp

    return access_denied, inaccessible


def check_rule(list_of_rules: dict, user_info: dict):
    access_denied = False
    inaccessible = {}

    for act in list_of_rules.keys():
        # list_of_rules[act]  # TODO how to handle ineq?
        info_type = list_of_rules[act]['info']
        info_value = list_of_rules[act]['value']
        info_op = list_of_rules[act]['operator']
        info_violation = list_of_rules[act]['violation']
        if info_type is None:
            continue
        actual_value = user_info[info_type]
        access_denied_ = not eval(f"actual_value {info_op} info_value")
        access_denied = access_denied or access_denied_
        if access_denied_:
            inaccessible[act] = info_violation

    return access_denied, inaccessible
