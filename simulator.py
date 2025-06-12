import generator as gen
import matplotlib.pyplot as plt
import numpy as np

def simulate_ADIP(modes, angles, parallel_modes_number, block_size, subset_size, samples_on, reuse_on, buffer_type = -1, global_buffer_type = -1):
    iterations = int(len(modes)/parallel_modes_number)
    global_samples_buffer = set()
    global_samples_buffer_list = [set() for i in range(iterations)]
    for index_x in range(0, block_size, subset_size):
        for index_y in range(0, block_size, subset_size):

            index = 0
            max_size = 0
            max_size_modes = []
            equations_constants_samples_buffer = global_samples_buffer.copy()

            list_of_samples_to_be_predicted = []
            list_of_modes = []
            for i in range(iterations):

                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]

                gen.calculate_samples(modes_subset, angles_subset, block_size, create_table=False)
                equations_constants_set = set()
                equations_constants_samples_set = set()
                equations_constants_reuse_map = {}
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        block_size,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x = index_x,
                        index_y = index_y,
                        subset_size = subset_size,
                        samples = samples_on,
                        reuse = reuse_on,
                        create_table = False)

                # gen.generate_sorted_equations_set(equations_constants_set, True)
                print("Subset:", index_x, index_y)
                print("Modes:", modes_subset)
                # for equation in equations_constants_samples_set:
                # print(equation)
                size = len(equations_constants_samples_set - (equations_constants_samples_buffer.union(global_samples_buffer_list[i])))
                print("Total N samples to be predicted:", len(equations_constants_samples_set))
                print("N samples to be predicted (with buffer): ",size)
                print("Local Buffer size", len(equations_constants_samples_buffer))
                print("Global Buffer size", len(global_samples_buffer_list[i]))
                list_of_samples_to_be_predicted.append(size)
                list_of_modes.append(str(modes_subset))

                match buffer_type:
                    case 0:
                        equations_constants_samples_buffer = equations_constants_samples_set.copy()
                    case 1:
                        equations_constants_samples_buffer = equations_constants_samples_buffer.union(equations_constants_samples_set)
                    case _:
                        pass

                match global_buffer_type:
                    case 0:
                        if index == 0:
                            global_samples_buffer = equations_constants_samples_buffer.copy()
                        else:
                            pass
                    case 1:
                        global_samples_buffer = global_samples_buffer.union(equations_constants_samples_buffer)
                    case 2:
                        global_samples_buffer_list[i] = equations_constants_samples_buffer.copy()
                    case _:
                        pass

                if max_size < size:
                    max_size = size
                    max_size_modes = modes_subset.copy()
                index += parallel_modes_number

            print("Most expensive modes:", max_size_modes, max_size)
            plt.rcParams['font.size'] = 4
            plt.figure(figsize=(12, 4))
            plt.bar(list_of_modes, list_of_samples_to_be_predicted)
            plt.savefig("graph_" + str(index_x) + "_" + str(index_y) + ".png", dpi=300,
                        bbox_inches='tight')



