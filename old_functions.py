def generate_control_sequence(sorted_equations_set, print_sequence):
    control_sequence = ["fetch", "fetch", "fetch", "fetch"]
    first_equation = sorted_equations_set[0]
    previous_value = get_reference_number(first_equation)

    for equation in sorted_equations_set[1:]:
        value = get_reference_number(equation)
        if value == previous_value:
            control_sequence.append("stall-calculate")
        elif value < previous_value:
            control_sequence.append("fetch-calculate")
            previous_value = value
        else:
            raise Exception("Equation set not sorted")

    control_sequence.append("calculate")

    if print_sequence:
        i = -4
        for instruction in control_sequence:
            i += 1
            print(i, instruction)