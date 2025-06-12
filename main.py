import generator as gen
import simulator as sim
import verifier as ver

path_input_modes = "./input/modes/"

option = 2
##modes = gen.all_modes
#input_modes = gen.all_modes
input_modes = [28,29,30,31,32,33,34,35,36,37,38,39,40]
parallel_modes_number = 3
buffer_type = 0
global_buffer_type = 2
block_size = 64
subset_size = 64
assert_equals = 1
normalize = False
heuristic_on = True
samples_on = True
reuse_on = True
n_average_fc = 16
n_average_fg = 16
n_samples = 1

def main(modes, control = -1):
    
    angles = gen.map_modes_to_angles(modes)

    filter_column_list, filter_column_list_normalized, filter_coefficient, filter_coefficient_normalized, filter_coefficients_set = gen.transform_coefficients(n_average_fc, n_average_fg, False, False )

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
            equations_constants_reuse_map = {}
            for mode, angle in zip(modes, angles):
                equations, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(mode, angle, block_size, "fc_heuristic", equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map, index_x = 0, index_y = 0, subset_size = 0, samples = samples_on, reuse = reuse_on, create_table = True)

        case 2:
            sim.simulate_ADIP(modes, angles, parallel_modes_number, block_size, subset_size, samples_on, reuse_on, buffer_type, global_buffer_type)
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
            print("Select a value for control between 0 and 7")
    

if __name__ == "__main__":
    main(input_modes, control = option)


