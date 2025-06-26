import generator as gen
import simulator as sim
import verifier as ver
from transform_block import TransformBlock

path_input_modes = "./input/modes/"

option = 2
input_modes = gen.all_modes
#parallel_modes_list = [5,24,2,4,5,25] #otimo 4x4
#parallel_modes_list = [16,9,3,3,2,3,5,24] #otimo 64x64
#parallel_modes_list = [len(input_modes)]
#parallel_modes_list = [1 for i in range(int(len(input_modes)))]
parallel_modes_list = [8, 12, 4, 4, 3, 3, 3, 4, 4, 12, 8]
buffer_type = -1
global_buffer_type = 1
block_size = 8
nTbW = 64
nTbH = 64
subset_size = 4
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
                equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(mode, angle, nTbW, nTbH, "fc_heuristic", equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map, index_x = 0, index_y = 0, subset_size = 0, refidx = 0, cidx = 0, samples = samples_on, reuse = reuse_on, create_table = True)

        case 2:
            sim.simulate_ADIP_IB(modes, angles, parallel_modes_list, nTbW, nTbH, subset_size, samples_on, reuse_on, refidx = 0, cidx = 0, buffer_type = buffer_type, global_buffer_type = global_buffer_type)
            #sim.simulate_Arq(modes, angles, parallel_modes_list, nTbW, nTbH, subset_size, samples_on, reuse_on, buffer_type, 0, 0)
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
        case 8:
                ver.verify_programmable_blocks_n8(0, [[1,2,4,8],[1,2,4,8],[-1,1],[-1,2]],filter_column_list_normalized)
                print("#####################################")
                ver.verify_programmable_blocks_n8(1, [[1,2,4,8,16,32,64], [1,2,4,8,16,32,64], [1,2,4,8,16,32,64], [-1, 1], [-1, 1], [1, 2]],filter_column_list_normalized)
                print("#####################################")
                ver.verify_programmable_blocks_n8(2, [[1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32],
                                                      [1, 2, 4, 8, 16, 32], [-1, 1], [0,1], [-1,1], [0,1], [0,1], [0,1]],
                                                  filter_column_list_normalized)
                print("#####################################")
                ver.verify_programmable_blocks_n8(3, [[1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32],
                                                      [1, 2, 4, 8, 16, 32], [-1, 1], [-1, 1], [0, 1]],
                                                  filter_column_list_normalized)
        case _:
            print("Select a value for control between 0 and 7")
    

if __name__ == "__main__":
    main(input_modes, control = option)


