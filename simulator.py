import generator as gen
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from cache import Cache as cache_c

def simulate_ADIP(modes, angles, parallel_modes_list, nTbW, nTbH, subset_size, samples_on, reuse_on, refidx = 0, cidx = 0, buffer_type = -1, global_buffer_type = -1):
    iterations = int(len(parallel_modes_list))
    global_samples_buffer_list = [set() for i in range(iterations)]
    list_of_samples_predictions = [[] for i in range(iterations)]
    table_of_samples_predictions = [[[0 for i in range(0, nTbH, subset_size)] for j in range(0, nTbW, subset_size)] for k in range(iterations)]

    column_sequences = [""]
    chunks = [modes[i:i + len(modes)] for i in range(0, len(modes), len(modes))]
    for chunk in chunks:
        column_sequences.append(str(chunk))

    table_of_mode_sequences = []
    columns = []
    global_samples_buffer = set()
    for index_x in range(0, nTbW, subset_size):
        columns.append(index_x)
        for index_y in range(0, nTbH, subset_size):
            sequence = [str(index_x) + ", " + str(index_y)]
            index = 0
            max_size = 0
            max_size_modes = []
            equations_constants_samples_buffer = set()

            list_of_samples_to_be_predicted = []
            list_of_modes = []
            for i in range(iterations):
                parallel_modes_number = parallel_modes_list[i]
                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]

                equations_constants_set = set()
                equations_constants_samples_set = set()
                equations_constants_reuse_map = {}
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        nTbW,
                        nTbH,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x=index_x,
                        index_y=index_y,
                        subset_size=subset_size,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)

                # gen.generate_sorted_equations_set(equations_constants_set, True)
                print("Subset:", index_x, index_y)
                print("Modes:", modes_subset)
                # for equation in equations_constants_samples_set:
                # print(equation)
                size = len(equations_constants_samples_set - (equations_constants_samples_buffer.union(global_samples_buffer)))
                print("Total N samples to be predicted:", len(equations_constants_samples_set))
                print("N samples to be predicted (with buffer): ",size)
                print("Local Buffer size", len(equations_constants_samples_buffer))
                print("Global Buffer size", len(global_samples_buffer))
                list_of_samples_to_be_predicted.append(size)
                list_of_modes.append(str(modes_subset))
                list_of_samples_predictions[i].append(size)
                table_of_samples_predictions[i][int(index_x/subset_size)][int(index_y/subset_size)] = size
                sequence.append(str(size))

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
                        global_samples_buffer = global_samples_buffer.union(equations_constants_samples_set)
                    case 2:
                        global_samples_buffer_list[i] = equations_constants_samples_buffer.copy()
                    case 3:
                        global_samples_buffer = equations_constants_samples_set.copy()
                    case 4:
                        global_samples_buffer_list[i] = global_samples_buffer_list[i].union(equations_constants_samples_buffer)
                    case _:
                        pass

                if max_size <= size:
                    max_size = size
                    max_size_modes = modes_subset.copy()
                index += parallel_modes_number

            print("Most expensive modes:", max_size_modes, max_size)
            table_of_mode_sequences.append(sequence)
            '''plt.rcParams['font.size'] = 4
            plt.figure(figsize=(12, 4))
            plt.bar(list_of_modes, list_of_samples_to_be_predicted)
            plt.savefig("graph_" + str(index_x) + "_" + str(index_y) + ".png", dpi=300,
                        bbox_inches='tight')'''

    total_sum = 0
    for column in table_of_samples_predictions:
        for values in column:
            column_sum = sum(values)
        total_sum += column_sum
    print(total_sum)

    list_of_max_values = []
    list_of_avg_values = []
    for preds in list_of_samples_predictions:
        list_of_max_values.append(max(preds))
        list_of_avg_values.append(sum(preds)/len(preds))

    '''plt.rcParams['font.size'] = 4
    plt.figure(figsize=(12, 4))
    plt.bar(list_of_modes, list_of_max_values)
    plt.savefig("graph_max_values_" + str(block_size) + ".png", dpi=300,
                bbox_inches='tight')

    plt.rcParams['font.size'] = 4
    plt.figure(figsize=(12, 4))
    plt.bar(list_of_modes, list_of_avg_values)
    plt.savefig("graph_avg_values_" + str(block_size) + ".png", dpi=300,
                bbox_inches='tight')'''

    table_index = 0
    for tables in table_of_samples_predictions:
        table_index += 1
        df = pd.DataFrame(list(zip(*tables)), columns=columns)
        excel_writer = pd.ExcelWriter(
            "table_samples_to_predict_" + str(nTbW) + "_" + str(nTbH) + "_" + str(table_index) + ".xlsx", engine='xlsxwriter')
        df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

        # Auto-adjust columns' width
        for column in df:
            column_width = 70
            col_iidx = df.columns.get_loc(column)
            excel_writer.sheets['equations'].set_column(col_iidx, col_iidx, column_width)

        excel_writer._save()

    table_of_mode_sequences_transposed = np.array(table_of_mode_sequences).T
    df = pd.DataFrame(list(zip(*table_of_mode_sequences_transposed)), columns=column_sequences)
    excel_writer = pd.ExcelWriter(
        "table_samples_sequence_" + str(nTbW) + "_" + str(nTbH) + ".xlsx", engine='xlsxwriter')
    df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

    # Auto-adjust columns' width
    for column in df:
        column_width = 70
        col_iidx = df.columns.get_loc(column)
        excel_writer.sheets['equations'].set_column(col_iidx, col_iidx, column_width)

    excel_writer._save()

def count_effective_and_reused_equations(cache, equations, mode):
    effective_equation_set = set()
    for data in cache.data_cache.keys():
        for column in equations:
            for equation in column:
                if data == equation:
                    effective_equation_set.add(equation)


    return len(effective_equation_set)

def simulate_ADIP_IB(modes, angles, parallel_modes_list, nTbW, nTbH, initial_index_x, initial_index_y, final_index_x, final_index_y, subset_size, samples_on, reuse_on, refidx = 0, cidx = 0, buffer_type = -1, global_buffer_type = -1):
    pred_cache = cache_c(int((nTbH/subset_size)/2))
    iterations = int(len(parallel_modes_list))
    buffer_size_list = []
    exit_buffer_dict = {}
    max_size = 0
    max_size_modes = []
    for mode in modes:
        exit_buffer_dict[mode] = 0

    global_samples_buffer_list = [set() for i in range(iterations)]
    list_of_samples_predictions = [[] for i in range(iterations)]
    table_of_samples_predictions = [[[0 for i in range(0, nTbH, subset_size)] for j in range(0, nTbW, subset_size)] for k in range(iterations)]

    column_sequences = [""]
    chunks = [modes[i:i + len(modes)] for i in range(0, len(modes), len(modes))]
    rotational_cache = [set() for i in range(3)]
    rotational_index = 0
    for chunk in chunks:
        column_sequences.append(str(chunk))

    table_of_mode_sequences = []
    columns = []
    global_samples_buffer = set()
    list_of_samples_to_be_predicted = []
    for index_x in range(initial_index_x, final_index_x, subset_size):
        columns.append(index_x)
        for index_y in range(initial_index_y, final_index_y, subset_size):
            sequence = [str(index_x) + ", " + str(index_y)]
            index = 0
            list_of_modes = []
            for i in range(iterations):
                parallel_modes_number = parallel_modes_list[i]
                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]

                equations_constants_set = set()
                equations_constants_samples_set = set()
                equations_constants_reuse_map = {}
                equations_list = []
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        nTbW,
                        nTbH,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x=index_x,
                        index_y=index_y,
                        subset_size=subset_size,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)
                    equations_list.append(equations)

                rot_size = 0
                rot_set = set()
                for cache in rotational_cache:
                    rot_set = rot_set.union(cache)
                    rot_size += len(cache)

                buffer_equations = set()
                buffer_size = 0
                match buffer_type:
                    case 0:
                        buffer_equations = global_samples_buffer
                        buffer_size = len(global_samples_buffer)
                        global_samples_buffer = global_samples_buffer.union(equations_constants_samples_set)
                    case 1:
                        buffer_equations = global_samples_buffer
                        buffer_size = len(global_samples_buffer)
                        if index_y == 0:
                            global_samples_buffer = equations_constants_samples_set.copy()
                    case 2:
                        buffer_equations = rot_set
                        buffer_size = rot_size
                        if index_y <= nTbH/2:
                            rotational_cache[rotational_index] = equations_constants_samples_set.copy()
                            rotational_index += 1
                        if rotational_index == 3:
                            rotational_index = 0
                    case 3:
                        buffer_equations = pred_cache.get_data()
                        buffer_size = len(pred_cache.get_data())
                        if index_y < nTbH/2:
                            pred_cache.decrement_lifetime()
                            #pred_cache.update(equations_constants_samples_set.copy())
                            pred_cache.insert(equations_constants_samples_set.copy())
                    case _:
                        pass

                # gen.generate_sorted_equations_set(equations_constants_set, True)
                print("Subset:", index_x, index_y)
                print("Modes:", modes_subset)
                for equation in equations_constants_samples_set:
                	print(equation)
                #size = len(equations_constants_samples_set - global_samples_buffer)
                size = len(equations_constants_samples_set - buffer_equations)
                print("Total N samples to be predicted:", len(equations_constants_samples_set))
                #print("N samples to be predicted (with cache): ", size)
                #print("Cache size", buffer_size)
                buffer_size_list.append(buffer_size)

                list_of_samples_to_be_predicted.append(size)
                list_of_modes.append(str(modes_subset))
                list_of_samples_predictions[i].append(size)
                table_of_samples_predictions[i][int(index_x/subset_size)][int(index_y/subset_size)] = size
                sequence.append(str(size))

                if max_size <= size:
                    max_size = size
                    max_size_modes = modes_subset.copy()
                index += parallel_modes_number

                for equation, mode in zip(equations_list, modes):
                    exit_buffer_size = count_effective_and_reused_equations(pred_cache, equation, mode)
                    if exit_buffer_dict[mode] < exit_buffer_size:
                        exit_buffer_dict[mode] = exit_buffer_size


            #print("Most expensive modes:", max_size_modes, max_size)
            table_of_mode_sequences.append(sequence)
            '''plt.rcParams['font.size'] = 4
            plt.figure(figsize=(12, 4))
            plt.bar(list_of_modes, list_of_samples_to_be_predicted)
            plt.savefig("graph_" + str(index_x) + "_" + str(index_y) + ".png", dpi=300,
                        bbox_inches='tight')'''

    total_sum = 0
    '''for column in table_of_samples_predictions:
        for values in column:
            column_sum = sum(values)
        total_sum += column_sum
    print(total_sum)'''
    print("Max size of equations to predict", max(list_of_samples_to_be_predicted))
    #print("Max size of cache", max(buffer_size_list))

    '''total_exit_buffer_size = 0
    for mode in modes:
        total_exit_buffer_size += exit_buffer_dict[mode]
        print("Max size of exit buffer for:", mode, exit_buffer_dict[mode])

    print("Total size", total_exit_buffer_size)'''

    list_of_max_values = []
    list_of_avg_values = []
    for preds in list_of_samples_predictions:
        list_of_max_values.append(max(preds))
        list_of_avg_values.append(sum(preds)/len(preds))

    '''plt.rcParams['font.size'] = 4
    plt.figure(figsize=(12, 4))
    plt.bar(list_of_modes, list_of_max_values)
    plt.savefig("graph_max_values_" + str(block_size) + ".png", dpi=300,
                bbox_inches='tight')

    plt.rcParams['font.size'] = 4
    plt.figure(figsize=(12, 4))
    plt.bar(list_of_modes, list_of_avg_values)
    plt.savefig("graph_avg_values_" + str(block_size) + ".png", dpi=300,
                bbox_inches='tight')'''

    '''table_index = 0
    for tables in table_of_samples_predictions:
        table_index += 1
        df = pd.DataFrame(list(zip(*tables)), columns=columns)
        excel_writer = pd.ExcelWriter(
            "table_samples_to_predict_" + str(nTbW) + "_" + str(nTbH) + "_" + str(buffer_type) + ".xlsx", engine='xlsxwriter')
        df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

        # Auto-adjust columns' width
        for column in df:
            column_width = 70
            col_iidx = df.columns.get_loc(column)
            excel_writer.sheets['equations'].set_column(col_iidx, col_iidx, column_width)

        excel_writer._save()'''

    '''table_of_mode_sequences_transposed = np.array(table_of_mode_sequences).T
    df = pd.DataFrame(list(zip(*table_of_mode_sequences_transposed)), columns=column_sequences)
    excel_writer = pd.ExcelWriter(
        "table_samples_sequence_" + str(nTbW) + "_" + str(nTbH) + ".xlsx", engine='xlsxwriter')
    df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

    # Auto-adjust columns' width
    for column in df:
        column_width = 70
        col_iidx = df.columns.get_loc(column)
        excel_writer.sheets['equations'].set_column(col_iidx, col_iidx, column_width)

    excel_writer._save()'''


def simulate_architecture(modes, angles, parallel_modes_list, number_of_units, nTbW, nTbH, subset_size, refidx, samples_on, reuse_on):
    iterations = int(len(parallel_modes_list))
    states_index = 0
    state_mapping = {}
    states_list = []
    for index_x in range(0, nTbW, subset_size):
        for index_y in range(0, nTbH, subset_size):
            equations_constants_set = set()
            equations_constants_samples_set = set()
            equations_constants_reuse_map = {}
            index = 0
            for i in range(iterations):
                unit_equation_mapping = {}
                unit_index = 0
                parallel_modes_number = parallel_modes_list[i]
                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]
                exit_buffer_unit_mapping = []
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        nTbW,
                        nTbH,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x=index_x,
                        index_y=index_y,
                        subset_size=subset_size,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)

                    for equations_column in equations:
                        for equation in equations_column:
                            if equation not in unit_equation_mapping.keys():
                                unit_equation_mapping[equation] = unit_index
                                unit_index += 1

                            exit_buffer_unit_mapping.append(unit_equation_mapping[equation])

                str_exit_buffer = str(exit_buffer_unit_mapping)
                match = False
                state_mapping_list = list(state_mapping.keys())
                states_index = 0
                while match == False and states_index < len(state_mapping_list):
                    if str_exit_buffer in state_mapping_list[states_index]:
                        states_index = state_mapping[str_exit_buffer]
                        match = True
                    else:
                        states_index += 1

                if not match:
                    state_mapping[str_exit_buffer] = states_index

                states_list.append(states_index)

                index += parallel_modes_number

    print(states_list)
    print(max(states_list))
    #print(states_index)
    #for output in state_mapping.keys():
    #    print(output)
    
def simulate_parallel_architecture_32x32(modes, angles, parallel_modes_list, number_of_units, refidx, samples_on, reuse_on):
    nTbW = 32
    nTbH = 32
    subset_size = 4
    iterations = int(len(parallel_modes_list))
    
    for initial_index_x in range(0, nTbW, 16):
    	for initial_index_y in range(0, nTbH, 16):
    		simulate_ADIP_IB(modes, angles, parallel_modes_list, nTbW, nTbH, initial_index_x, initial_index_y, initial_index_x + 16, initial_index_y + 16, subset_size, samples_on, reuse_on, refidx = 0, cidx = 0, buffer_type = -1, global_buffer_type = -1)
    
    states_index = 0
    states_list_index = -1
    state_mapping = {}
    states_list = []
    for index_x in range(0, 16, subset_size):
        for index_y in range(0, 16, subset_size):
            equations_constants_set = set()
            equations_constants_samples_set = set()
            equations_constants_reuse_map = {}
            index = 0
            states_list_index += 1
            for i in range(iterations):
                unit_equation_mapping = {}
                unit_index = 0
                parallel_modes_number = parallel_modes_list[i]
                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]
                exit_buffer_unit_mapping = []
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        nTbW,
                        nTbH,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x=index_x,
                        index_y=index_y,
                        subset_size=subset_size,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)

                    for equations_column in equations:
                        for equation in equations_column:
                            if equation not in unit_equation_mapping.keys():
                                unit_equation_mapping[equation] = unit_index
                                unit_index += 1

                            exit_buffer_unit_mapping.append(unit_equation_mapping[equation])

                str_exit_buffer = str(exit_buffer_unit_mapping)
                match = False
                state_mapping_list = list(state_mapping.keys())
                states_index = 0
                while match == False and states_index < len(state_mapping_list):
                    if str_exit_buffer in state_mapping_list[states_index]:
                        states_index = state_mapping[str_exit_buffer]
                        match = True
                    else:
                        states_index += 1

                if not match:
                    state_mapping[str_exit_buffer] = states_index

                states_list.append(states_index)

                index += parallel_modes_number

    print(states_list)
    #print(states_index)
    #for output in state_mapping.keys():
    #    print(output)
    
    state_mapping = {}
    states_list = []
    for index_x in range(16, 32, subset_size):
        for index_y in range(0, 16, subset_size):
            equations_constants_set = set()
            equations_constants_samples_set = set()
            equations_constants_reuse_map = {}
            index = 0
            states_list_index += 1
            for i in range(iterations):
                unit_equation_mapping = {}
                unit_index = 0
                parallel_modes_number = parallel_modes_list[i]
                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]
                exit_buffer_unit_mapping = []
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        nTbW,
                        nTbH,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x=index_x,
                        index_y=index_y,
                        subset_size=subset_size,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)

                    for equations_column in equations:
                        for equation in equations_column:
                            if equation not in unit_equation_mapping.keys():
                                unit_equation_mapping[equation] = unit_index
                                unit_index += 1

                            exit_buffer_unit_mapping.append(unit_equation_mapping[equation])

                str_exit_buffer = str(exit_buffer_unit_mapping)
                match = False
                state_mapping_list = list(state_mapping.keys())
                states_index = 0
                while match == False and states_index < len(state_mapping_list):
                    if str_exit_buffer in state_mapping_list[states_index]:
                        states_index = state_mapping[str_exit_buffer]
                        match = True
                    else:
                        states_index += 1

                if not match:
                    state_mapping[str_exit_buffer] = states_index

                states_list.append(states_index)

                index += parallel_modes_number

    print(states_list)
    #print(states_index)
    #for output in state_mapping.keys():
    #    print(output)
    
    state_mapping = {}
    states_list = []
    for index_x in range(0, 16, subset_size):
        for index_y in range(16, 32, subset_size):
            equations_constants_set = set()
            equations_constants_samples_set = set()
            equations_constants_reuse_map = {}
            index = 0
            states_list_index += 1
            for i in range(iterations):
                unit_equation_mapping = {}
                unit_index = 0
                parallel_modes_number = parallel_modes_list[i]
                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]
                exit_buffer_unit_mapping = []
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        nTbW,
                        nTbH,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x=index_x,
                        index_y=index_y,
                        subset_size=subset_size,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)

                    for equations_column in equations:
                        for equation in equations_column:
                            if equation not in unit_equation_mapping.keys():
                                unit_equation_mapping[equation] = unit_index
                                unit_index += 1

                            exit_buffer_unit_mapping.append(unit_equation_mapping[equation])

                str_exit_buffer = str(exit_buffer_unit_mapping)
                match = False
                state_mapping_list = list(state_mapping.keys())
                states_index = 0
                while match == False and states_index < len(state_mapping_list):
                    if str_exit_buffer in state_mapping_list[states_index]:
                        states_index = state_mapping[str_exit_buffer]
                        match = True
                    else:
                        states_index += 1

                if not match:
                    state_mapping[str_exit_buffer] = states_index

                states_list.append(states_index)

                index += parallel_modes_number

    print(states_list)
    #print(states_index)
    #for output in state_mapping.keys():
    #    print(output)
    
    state_mapping = {}
    states_list = []
    for index_x in range(16, 32, subset_size):
        for index_y in range(16, 32, subset_size):
            equations_constants_set = set()
            equations_constants_samples_set = set()
            equations_constants_reuse_map = {}
            index = 0
            states_list_index += 1
            for i in range(iterations):
                unit_equation_mapping = {}
                unit_index = 0
                parallel_modes_number = parallel_modes_list[i]
                modes_subset = modes[index:index + parallel_modes_number]
                angles_subset = angles[index:index + parallel_modes_number]
                exit_buffer_unit_mapping = []
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        nTbW,
                        nTbH,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x=index_x,
                        index_y=index_y,
                        subset_size=subset_size,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)

                    for equations_column in equations:
                        for equation in equations_column:
                            if equation not in unit_equation_mapping.keys():
                                unit_equation_mapping[equation] = unit_index
                                unit_index += 1

                            exit_buffer_unit_mapping.append(unit_equation_mapping[equation])

                str_exit_buffer = str(exit_buffer_unit_mapping)
                match = False
                state_mapping_list = list(state_mapping.keys())
                states_index = 0
                while match == False and states_index < len(state_mapping_list):
                    if str_exit_buffer in state_mapping_list[states_index]:
                        states_index = state_mapping[str_exit_buffer]
                        match = True
                    else:
                        states_index += 1

                if not match:
                    state_mapping[str_exit_buffer] = states_index

                states_list.append(states_index)

                index += parallel_modes_number

    print(states_list)
    #print(states_index)
    #for output in state_mapping.keys():
    #    print(output)
    
    
def simulate_parallel_architecture_64x64(modes, angles, parallel_modes_list, number_of_units, refidx, samples_on, reuse_on):
    nTbW = 64
    nTbH = 64
    subset_size = 4
    iterations = int(len(parallel_modes_list))
    
    for initial_index_x in range(0, nTbW, 16):
    	for initial_index_y in range(0, nTbH, 16):
   	
    		simulate_ADIP_IB(modes, angles, parallel_modes_list, nTbW, nTbH, initial_index_x, initial_index_y, initial_index_x + 16, initial_index_y + 16, subset_size, samples_on, reuse_on, refidx = 0, cidx = 0, buffer_type = -1, global_buffer_type = -1)

def simulate_Arq(modes, angles, parallel_modes_list, nTbW, nTbH, subset_size, samples_on, reuse_on, buffer_type = -1, refidx = 0, cidx = 0):
    iterations = int(len(parallel_modes_list))
    samples_buffer_list = [[set() for i in range(0, nTbH, subset_size)] for j in range(0, nTbW, subset_size)]
    samples_size_list = [[0 for i in range(0, nTbH, subset_size)] for j in range(0, nTbW, subset_size)]
    samples_buffer_cache = set()
    divided_cache = [set() for i in  range(0, nTbH, subset_size)]
    for index_x in range(0, nTbW, subset_size):
        index = 0
        list_of_samples_to_be_predicted = []
        list_of_modes = []
        cache_prev_cycle_set = samples_buffer_cache
        for i in range(iterations):
            parallel_modes_number = parallel_modes_list[i]
            modes_subset = modes[index:index + parallel_modes_number]
            angles_subset = angles[index:index + parallel_modes_number]
            for index_y in range(0, nTbH, subset_size):
                equations_constants_set = set()
                equations_constants_samples_set = set()
                equations_constants_reuse_map = {}
                for mode, angle in zip(modes_subset, angles_subset):
                    equations, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = gen.calculate_equations(
                        mode,
                        angle,
                        nTbW,
                        nTbH,
                        "fc_heuristic",
                        equations_constants_set,
                        equations_constants_samples_set,
                        equations_constants_reuse_map,
                        index_x=index_x,
                        index_y=index_y,
                        subset_size=subset_size,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)

                size = len(equations_constants_samples_set - (
                    samples_buffer_list[int(index_x / subset_size)][int(index_y / subset_size)]))
                samples_buffer_cache = samples_buffer_cache.union(equations_constants_samples_set)
                size_divided_cache  = len(equations_constants_samples_set - divided_cache[int(index_y/subset_size)])
                divided_cache[int(index_y/subset_size)] = divided_cache[int(index_y/subset_size)].union(equations_constants_samples_set)
                size_cache = len(equations_constants_samples_set - cache_prev_cycle_set)
                samples_size_list[int(index_x / subset_size)][int(index_y / subset_size)] = size_cache

                match buffer_type:
                    case 0:
                        samples_buffer_list[int(index_x/subset_size)][int(index_y/subset_size)] = equations_constants_samples_set.copy()
                    case 1:
                        samples_buffer_list[int(index_x / subset_size)][int(index_y / subset_size)] = equations_constants_samples_set.copy().union(samples_buffer_list[int(index_x / subset_size)][int(index_y / subset_size)])
                    case _:
                        pass

                # gen.generate_sorted_equations_set(equations_constants_set, True)
                print("Subset:", index_x, index_y)
                print("Modes:", modes_subset)
                # for equation in equations_constants_samples_set:
                # print(equation)
                print("Total N samples to be predicted:", len(equations_constants_samples_set))
                print("N samples to be predicted (with buffer): ", size)
                print("N samples to be predicted (with cache): ", size_cache)
                print("N samples to be predicted (with divided cache): ", size_divided_cache)
                list_of_samples_to_be_predicted.append(size)
                list_of_modes.append(str(modes_subset))

            index += parallel_modes_number

    samples_size_list_transposed = np.array(samples_size_list).T
    list_of_max_sizes = []
    for samples_size in samples_size_list_transposed:
        list_of_max_sizes.append(int(max(samples_size)))
    print(list_of_max_sizes)
