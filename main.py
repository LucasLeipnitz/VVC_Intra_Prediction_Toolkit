import generator as gen
import simulator as sim
import verifier as ver
from transform_block import TransformBlock

path_input_modes = "./input/modes/"

option = 8
input_modes = gen.all_modes
#input_modes = [34,35,37]
#input_modes = [2,3,4,5,6,7]
#parallel_modes_list = [2,2,2]
#input_modes = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] #Waip
#input_modes = [53,54,55,56,57,58,59,60,61,62,63,64,65,66] #Waip
#parallel_modes_list = [5,24,2,4,5,25] #otimo 4x4
#parallel_modes_list = [16,9,3,3,2,3,5,24] #otimo 64x64
#parallel_modes_list = [len(input_modes)]
#parallel_modes_list = [1 for i in range(int(len(input_modes)))]
#parallel_modes_list = [8, 12, 3, 2, 3, 2, 1, 3, 1, 2, 3, 2, 3, 12, 8] #para 32x32
#parallel_modes_list = [6, 6, 8, 3, 3, 2, 3, 2, 1, 3, 2, 3, 3, 8, 6, 6] #para 16x16
parallel_modes_list = [20, 9, 4, 3, 9, 20] #para 4x4

buffer_type = -1
global_buffer_type = 1
block_size = 64
nTbW = 4
nTbH = 4
subset_size_x = 4
subset_size_y = 4
#subset_size_y = subset_size_x #blocos quadrados
assert_equals = 0
normalize = False	
heuristic_on = True
samples_on = True
reuse_on = False
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
                equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(mode, angle, nTbW, nTbH, "fc_heuristic", equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map, index_x = 0, index_y = 0, subset_size_x = subset_size_x, subset_size_y = subset_size_y, refidx = 0, cidx = 0, samples = samples_on, reuse = reuse_on, create_table = True)

        case 2:
            sim.simulate_ADIP_IB(modes, angles, parallel_modes_list, nTbW, nTbH, 0, 0, 4, 4, subset_size_x, subset_size_y, samples_on, reuse_on, refidx = 0, cidx = 0, buffer_type = buffer_type, global_buffer_type = global_buffer_type)
            #sim.simulate_number_of_states(modes, angles, parallel_modes_list, nTbW, nTbH, 0, 0, 4, 4, subset_size_x, subset_size_y, refidx = 0, samples_on = samples_on , reuse_on = reuse_on)
            #sim.simulate_parallel_architecture_32x32(modes, angles, parallel_modes_list, 120, refidx = 0, samples_on = samples_on , reuse_on = reuse_on)
            #sim.simulate_parallel_architecture_64x64(modes, angles, parallel_modes_list, 120, refidx = 0, samples_on = samples_on , reuse_on = reuse_on)
            #sim.simulate_list_of_states(modes, angles, parallel_modes_list, nTbW, nTbH, 0, 0, 4, 4, subset_size_x, subset_size_y, refidx = 0, samples_on = samples_on , reuse_on = reuse_on)
            gen.angular_input_mapping(modes, angles, parallel_modes_list, nTbW, nTbH, 0, 0, 4, 4, subset_size_x, subset_size_y, refidx = 0, samples_on = samples_on , reuse_on = reuse_on, coefficients_table = filter_column_list)
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
            leaf_mapping = {}
            sim.QT_split(64,64, leaf_mapping)
            print(leaf_mapping)
            print(16,16)
            number_of_blocks = sim.MTT_split(64, 64, (0,0))
            print(number_of_blocks)
        case 8:
            top_samples, left_samples = gen.generate_samples_buffer(196, 19, 18)
            print(top_samples)
            print(left_samples)
            tb = TransformBlock(4, 4, modes[0], angles[0], 0, 4 * 2 + 2, 4 * 2 + 2, 0)
            tb.calculate_pred_values()
            f = open("output_results.txt", "r")
            pred_result = ver.generate_output(f, tb, angles[0], 0, 0, 4, 4, top_samples, left_samples, "fc_heuristic")
            print(pred_result)
            result_index = 0
            for line in f:
                if line != "#\n" and line != "UUUUUUUU\n":
                    if int(line, 2) == pred_result[result_index]:
                        print("Passed")
                    else:
                        print("Not passed: ", int(line, 2), pred_result[result_index])

                    result_index += 1
        case _:
            print("Select a value for control between 0 and 7")
    

if __name__ == "__main__":
    main(input_modes, control = option)


