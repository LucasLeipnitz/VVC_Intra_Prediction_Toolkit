import generator as gen
import verifier as ver

path_input_modes = "./input/modes/"

option = 2
##modes = gen.all_modes
input_modes = gen.all_modes
parallel_modes_number = 5
block_size = 4
assert_equals = 1
normalize = False
heuristic_on = True
reuse_on = True
n_average_fc = 16
n_average_fg = 16
n_samples = 1

def main(modes, control = -1):
    
    angles = gen.map_modes_to_angles(modes)

    filter_column_list, filter_column_list_normalized, filter_coefficient, filter_coefficient_normalized, filter_coefficients_set = gen.transform_coefficients(n_average_fc, n_average_fg, False, False)

    if n_samples >= 4:
        filter_column_list = gen.generate_coefficients_for_parallel_prediction(filter_column_list, n_samples, True)
        filter_column_list_normalized = gen.generate_coefficients_for_parallel_prediction(filter_column_list_normalized, n_samples, False)

    match control:
        case 0:
            gen.calculate_samples(modes, angles, block_size, normalize = normalize, create_table = True)
        case 1:
            gen.calculate_samples(modes, angles, block_size, normalize = normalize, create_table = False)
            equations_constants_set = set()
            equations_constants_samples_set = set()
            for mode, angle in zip(modes, angles):
                equations, equations_constants_set, equations_constants_samples_set = gen.calculate_equations(mode, angle, block_size, "fc_heuristic", equations_constants_set, equations_constants_samples_set, reuse_on, create_table = True)

        case 2:
            index = 0
            max_size = 0
            max_size_modes = []
            while index < len(modes):

                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]

                gen.calculate_samples(modes_subset, angles_subset, block_size, normalize=normalize, create_table=False)
                equations_constants_set = set()
                equations_constants_samples_set = set()
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_set, equations_constants_samples_set = gen.calculate_equations(mode,
                                                                                                                  angle,
                                                                                                                  block_size,
                                                                                                                  "fc_heuristic",
                                                                                                                  equations_constants_set,
                                                                                                                  equations_constants_samples_set,
                                                                                                                  reuse_on,
                                                                                                                  create_table = False)
                #gen.generate_sorted_equations_set(equations_constants_set, True)
                print(modes_subset)
                for equation in equations_constants_samples_set:
                    print(equation)
                size = len(equations_constants_samples_set)
                print(size)
                if max_size < size:
                    max_size = size
                    max_size_modes = modes_subset.copy()
                index += parallel_modes_number
            print(max_size)
            print(max_size_modes)
        case 3:
            gen.calculate_iidx_ifact(modes, angles, block_size, heuristic_on, n_average_fc)
        case 4:
            for mode, angle in zip(modes, angles):
                equations, equations_set = gen.calculate_equations(mode, angle, block_size, "fc_heuristic", 1)
                sorted_equations_set = gen.generate_sorted_equations_set(equations_set, True)
                gen.generate_control_sequence(sorted_equations_set,True)
        case 5:
            gen.generate_mcm_blocks(filter_column_list_normalized)
            input_map = gen.generate_port_mapping(filter_column_list_normalized, n_samples)
            if n_samples < 4:
                gen.generate_mux(filter_column_list, input_map)
            else:
                gen.generate_mux_n_samples(filter_column_list, input_map, n_samples)
        case 6:
            gen.generate_rom(filter_column_list, filter_coefficients_set)
        case 7:
            if assert_equals:
                test_input = ver.automated_tests(196, -66, 38)
                ver.assert_equals_planar(test_input, 32)
                ver.assert_equals_dc(test_input, 32, 135)
                for i in range (0,16):
                    ver.assert_equals_angular(test_input, modes[i], angles[i], block_size, -66, 38)
            else:
                ver.automated_tests(196, -66, 38)

        case _:
            print("Select a value for control between 0 and 3")
    

if __name__ == "__main__":
    main(input_modes, control = option)


