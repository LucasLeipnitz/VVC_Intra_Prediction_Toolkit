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

def generate_programmable_blocks_n8(block, control_variables_set_list, filter_coefficient):
    match block:
        case 0:
            for coefficient in filter_coefficient[0]:
                match_coefficient = False
                match_times = 0
                control_sequences = []
                for A in control_variables_set_list[0]:
                    for B in control_variables_set_list[1]:
                        for C in control_variables_set_list[2]:
                            for D in control_variables_set_list[3]:
                                for E in control_variables_set_list[4]:
                                    x = A + B*C
                                    if coefficient == (x*(1-D) + D*1)*E:
                                        control_sequence_string = (
                                            str(bin(control_variables_set_list[0].index(A))[2:]) + str(
                                            bin(control_variables_set_list[1].index(B))[2:]) + str(
                                            bin(control_variables_set_list[2].index(C))[2:]) + str(
                                            bin(control_variables_set_list[3].index(D))[2:]) + str(
                                            bin(control_variables_set_list[4].index(E))[2:]))
                                        control_sequences.append(control_sequence_string)
                                        match_coefficient = True
                                        match_times += 1

                if match_coefficient:
                    print("Coefficient match:", coefficient)
                    print(control_sequences[0])
                else:
                    print("Coefficient doesn't match:", coefficient)
        case 1:
            A_values = set()
            B_values = set()
            C_values = set()
            for coefficient in filter_coefficient[1]:
                match_coefficient = False
                match_times = 0
                control_sequences = []
                for A in control_variables_set_list[0]:
                    for B in control_variables_set_list[1]:
                        for C in control_variables_set_list[2]:
                            for D in control_variables_set_list[3]:
                                for E in control_variables_set_list[4]:
                                    for F in control_variables_set_list[5]:
                                        if coefficient == (A + (B*D + C)*E)*F and not match_times:
                                            A_values.add(A)
                                            B_values.add(B)
                                            C_values.add(C)
                                            control_sequence_string = (
                                                str(bin(control_variables_set_list[0].index(A))[2:]) + str(
                                                bin(control_variables_set_list[1].index(B))[2:]) + str(
                                                bin(control_variables_set_list[2].index(C))[2:]) + str(
                                                bin(control_variables_set_list[3].index(D))[2:]) + str(
                                                bin(control_variables_set_list[4].index(E))[2:]) + str(
                                                bin(control_variables_set_list[5].index(F))[2:]))
                                            control_sequences.append(control_sequence_string)
                                            match_coefficient = True
                                            match_times += 1

                if match_coefficient:
                    print("Coefficient match:", coefficient)
                    print(control_sequences[0])
                else:
                    print("Coefficient doesn't match:", coefficient)
            print(A_values)
            print(B_values)
            print(C_values)

        case 2:
            A_values = set()
            B_values = set()
            #C_values = set()
            for coefficient in filter_coefficient[2]:
                match_coefficient = False
                match_times = 0
                control_sequences = []
                for A in control_variables_set_list[0]:
                    for B in control_variables_set_list[1]:
                        for D in control_variables_set_list[2]:
                            for K in control_variables_set_list[3]:
                                for F in control_variables_set_list[4]:
                                    for G in control_variables_set_list[5]:
                                        for H in control_variables_set_list[6]:
                                            x = B+1*D
                                            y = A + (K*x*8 + (1-K)*x)*F
                                            z = G*y + (1-G)*y*2
                                            out = H*z + (1-H)*x*8
                                            if coefficient == out and not match_times:
                                                A_values.add(A)
                                                B_values.add(B)
                                                #C_values.add(C)
                                                control_sequence_string = (
                                                        str(bin(control_variables_set_list[0].index(A))[
                                                            2:]) + str(
                                                    bin(control_variables_set_list[1].index(B))[2:]) + str(
                                                    bin(control_variables_set_list[2].index(D))[2:]) + str(
                                                    bin(control_variables_set_list[3].index(K))[2:]) + str(
                                                    bin(control_variables_set_list[4].index(F))[2:]) + str(
                                                    bin(control_variables_set_list[5].index(G))[2:]) + str(
                                                    bin(control_variables_set_list[6].index(H))[2:]))
                                                control_sequences.append(control_sequence_string)
                                                match_coefficient = True
                                                match_times += 1

                if match_coefficient:
                    print("Coefficient match:", coefficient)
                    print(control_sequences[0])
                else:
                    print("Coefficient doesn't match:", coefficient)
            print(A_values)
            print(B_values)
            #print(C_values)
        case 3:
            A_values = set()
            B_values = set()
            C_values = set()
            for coefficient in filter_coefficient[3]:
                match_coefficient = False
                match_times = 0
                control_sequences = []
                for A in control_variables_set_list[0]:
                    for B in control_variables_set_list[1]:
                        for C in control_variables_set_list[2]:
                            for D in control_variables_set_list[3]:
                                for K in control_variables_set_list[4]:
                                    for E in control_variables_set_list[5]:
                                        for F in control_variables_set_list[6]:
                                            x = B + C * D
                                            y = K*x + (1-K)*1
                                            out = F*(A - x) + (1-F)*y*E
                                            if coefficient == out and not match_times:
                                                A_values.add(A)
                                                B_values.add(B)
                                                C_values.add(C)
                                                control_sequence_string = (
                                                    str(bin(control_variables_set_list[0].index(A))[2:]) + str(
                                                    bin(control_variables_set_list[1].index(B))[2:]) + str(
                                                    bin(control_variables_set_list[2].index(C))[2:]) + str(
                                                    bin(control_variables_set_list[3].index(D))[2:]) + str(
                                                    bin(control_variables_set_list[4].index(K))[2:]) + str(
                                                    bin(control_variables_set_list[5].index(E))[2:]) + str(
                                                    bin(control_variables_set_list[6].index(F))[2:]))
                                                control_sequences.append(control_sequence_string)
                                                match_coefficient = True
                                                match_times += 1

                if match_coefficient:
                    print("Coefficient match:", coefficient)
                    print(control_sequences[0])
                else:
                    print("Coefficient doesn't match:", coefficient)
            print(A_values)
            print(B_values)
            print(C_values)
        case _:
            print("Select a value for block between 0 and 3")

def verify_programmable_blocks_n8(block, input_sequence, result_sequence, control_sequence, control_variables_set_list):
    match block:
        case 0:
            for input_ref, result, control_signal in zip(input_sequence, result_sequence, control_sequence):
                A = int(control_signal[0],2)
                B = int(control_signal[1],2)
                C = int(control_signal[2],2)
                D = int(control_signal[3],2)
                E = int(control_signal[4],2)

                x = control_variables_set_list[0][A]*input_ref + control_variables_set_list[1][B]*control_variables_set_list[2][C]*input_ref
                final_value = (x*(1-control_variables_set_list[3][D]) + control_variables_set_list[3][D]*input_ref)*control_variables_set_list[4][E]
                if result == final_value:
                    print("Passed")
                else:
                    print(result,"!=",final_value)

        case _:
            print("Select a value for block between 0 and 3")

def generate_output(f, base, angle, size, input, filterFlag):
    x_base, y_base = base[0], base[1]
    for y in range(y_base, y_base + size):
        iIdx = ((y + 1) * angle) >> 5
        iFact = ((y + 1) * angle) & 31
        f.write(str(calculate_pred_y(input, x_base, iIdx, iFact, filterFlag)) + "\n")
        # print(str(calculate_pred_y(input,x_base,iIdx,iFact, filterFlag)) + "\n")