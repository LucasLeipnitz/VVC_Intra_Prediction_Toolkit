import numpy as np
import pandas as pd
import math as mh
import re
import random as rm
from collections import defaultdict

from numpy.f2py.auxfuncs import throw_error

from transform_block import TransformBlock

path = "./"
path_output_metrics = path + "output/metrics/"
path_output_blocks = path + "output/mcm_block/"
path_output_states = path + "output/states/"
path_values = path + "output/values_iidx_ifact/"
path_samples = path + "output/samples/"
path_planar = path + "output/planar/"
path_equations = "output/equations/"

all_modes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
             58, 59, 60, 61, 62, 63, 64, 65, 66]
modes1 = [2,3,4,5,6,7,8,9,10,11,12,13]
modes2 = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
          61, 62, 63, 64, 65, 66]
modes3 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34]
'''modes3 = [2, 3, 7, 10, 18, 23, 26, 30, 33, 34, 35, 43, 46, 49, 50, 54]
modes4 = [35]
modes5 = [35, 54]
modes_positive = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
modes_negative = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
angles1 = [-32, -29, -26, -23, -20, -18, -16, -14, -12, -10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12, 14,
           16, 18, 20, 23, 26, 29, 32]
angles2 = [32, 29, 26, 23, 20, 18, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1, 0, -1, -2, -3, -4, -6, -8, -10, -12, -14, -16, -18,
           -20, -23, -26, -29, -32]
angles3 = [32, 29, 18, 12, 0, -6, -12, -20, -29, -32, -29, -10, -4, -1, 0, 4]
angles4 = [-29]
angles5 = [-32, -29]
all_angles = [32, 29, 26, 23, 20, 18, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1, 0, -1, -2, -3, -4, -6, -8, -10, -12, -14, -16,
              -18, -20, -23, -26, -29, -32, -29, -26, -23, -20, -18, -16, -14, -12, -10, -8, -6, -4, -3, -2, -1, 0, 1,
              2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32]
angles_positive = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32]
angles_negative = [-32, -29, -26, -23, -20, -18, -16, -14, -12, -10, -8, -6, -4, -3, -2, -1, 0]'''

angles_map = {
    2: 32,
    3: 29,
    4: 26,
    5: 23,
    6: 20,
    7: 18,
    8: 16,
    9: 14,
    10: 12,
    11: 10,
    12: 8,
    13: 6,
    14: 4,
    15: 3,
    16: 2,
    17: 1,
    18: 0,
    19: -1,
    20: -2,
    21: -3,
    22: -4,
    23: -6,
    24: -8,
    25: -10,
    26: -12,
    27: -14,
    28: -16,
    29: -18,
    30: -20,
    31: -23,
    32: -26,
    33: -29,
    34: -32,
    35: -29,
    36: -26,
    37: -23,
    38: -20,
    39: -18,
    40: -16,
    41: -14,
    42: -12,
    43: -10,
    44: -8,
    45: -6,
    46: -4,
    47: -3,
    48: -2,
    49: -1,
    50: 0,
    51: 1,
    52: 2,
    53: 3,
    54: 4,
    55: 6,
    56: 8,
    57: 10,
    58: 12,
    59: 14,
    60: 16,
    61: 18,
    62: 20,
    63: 23,
    64: 26,
    65: 29,
    66: 32
}

fc_coefficients = {
    "0[0]": 0,
    "1[0]": -1,
    "2[0]": -2,
    "3[0]": -2,
    "4[0]": -2,
    "5[0]": -3,
    "6[0]": -4,
    "7[0]": -4,
    "8[0]": -4,
    "9[0]": -5,
    "10[0]": -6,
    "11[0]": -6,
    "12[0]": -6,
    "13[0]": -5,
    "14[0]": -4,
    "15[0]": -4,
    "16[0]": -4,
    "17[0]": -4,
    "18[0]": -4,
    "19[0]": -4,
    "20[0]": -4,
    "21[0]": -3,
    "22[0]": -2,
    "23[0]": -2,
    "24[0]": -2,
    "25[0]": -2,
    "26[0]": -2,
    "27[0]": -2,
    "28[0]": -2,
    "29[0]": -1,
    "30[0]": 0,
    "31[0]": 0,
    "0[1]": 64,
    "1[1]": 63,
    "2[1]": 62,
    "3[1]": 60,
    "4[1]": 58,
    "5[1]": 57,
    "6[1]": 56,
    "7[1]": 55,
    "8[1]": 54,
    "9[1]": 53,
    "10[1]": 52,
    "11[1]": 49,
    "12[1]": 46,
    "13[1]": 44,
    "14[1]": 42,
    "15[1]": 39,
    "16[1]": 36,
    "17[1]": 33,
    "18[1]": 30,
    "19[1]": 29,
    "20[1]": 28,
    "21[1]": 24,
    "22[1]": 20,
    "23[1]": 18,
    "24[1]": 16,
    "25[1]": 15,
    "26[1]": 14,
    "27[1]": 12,
    "28[1]": 10,
    "29[1]": 7,
    "30[1]": 4,
    "31[1]": 2,
    "0[2]": 0,
    "0[3]": 0
}

fg_coefficients = {
    "0[0]": 16,
    "1[0]": 16,
    "2[0]": 15,
    "3[0]": 15,
    "4[0]": 14,
    "5[0]": 14,
    "6[0]": 13,
    "7[0]": 13,
    "8[0]": 12,
    "9[0]": 12,
    "10[0]": 11,
    "11[0]": 11,
    "12[0]": 10,
    "13[0]": 10,
    "14[0]": 9,
    "15[0]": 9,
    "16[0]": 8,
    "17[0]": 8,
    "18[0]": 7,
    "19[0]": 7,
    "20[0]": 6,
    "21[0]": 6,
    "22[0]": 5,
    "23[0]": 5,
    "24[0]": 4,
    "25[0]": 4,
    "26[0]": 3,
    "27[0]": 3,
    "28[0]": 2,
    "29[0]": 2,
    "30[0]": 1,
    "31[0]": 1,
    "0[1]": 32,
    "1[1]": 32,
    "2[1]": 31,
    "3[1]": 31,
    "4[1]": 30,
    "5[1]": 30,
    "6[1]": 29,
    "7[1]": 29,
    "8[1]": 28,
    "9[1]": 28,
    "10[1]": 27,
    "11[1]": 27,
    "12[1]": 26,
    "13[1]": 26,
    "14[1]": 25,
    "15[1]": 25,
    "16[1]": 24,
    "17[1]": 24,
    "18[1]": 23,
    "19[1]": 23,
    "20[1]": 22,
    "21[1]": 22,
    "22[1]": 21,
    "23[1]": 21,
    "24[1]": 20,
    "25[1]": 20,
    "26[1]": 19,
    "27[1]": 19,
    "28[1]": 18,
    "29[1]": 18,
    "30[1]": 17,
    "31[1]": 17,
    "0[2]": 16,
    "0[3]": 0,
    "1[2]": 16,
    "1[3]": 0
}

fc_heuristic = {}
fg_heuristic = {}
ft_coefficients = {"fc": fc_coefficients, "fg": fg_coefficients, "fc_heuristic": fc_heuristic,
                   "fg_heuristic": fg_heuristic}
intraHorVerDistThres = {2: 24, 3: 14, 4: 2, 5: 0, 6: 0}

def map_modes_to_angles(modes):
    angles = []
    for mode in modes:
        angles.append(angles_map[mode])

    return angles

def simmetry_rule(p, index, coef_table):
    if (coef_table == "fc"):
        return str(32 - int(p)), str(3 - int(index))
    elif (coef_table == "fg"):
        return str(33 - int(p)), str(3 - int(index))
    else:
        raise Exception("Coefficient table unknow: " + coef_table)


def get_coefficient_value(table, coefficient):
    if (coefficient not in ft_coefficients[table]):
        p, index = re.findall(r'\d+',
                              coefficient)  # get p[index] from string containing and put it in two separately variables
        p, index = simmetry_rule(p, index,
                                 table)  # transform in a value that exists in the coefficients by the simmetry rule
        return ft_coefficients[table][p + '[' + index + ']']
    else:
        return ft_coefficients[table][coefficient]


def calculate_iidx_ifact(modes, angles, size, heuristic_on, n_average_fc):
    values_ifact = []
    values_iidx = []
    columns = []
    for i, j in zip(modes, angles):
        tb = TransformBlock(size, size, i, j, 0, size * 2 + 2, size * 2 + 2, 0)
        tb.calculate_constants_mode()
        columns.append(i)
        values_iidx.append(tb.array_iidx.copy())
        if heuristic_on:
            values_ifact.append((np.array(tb.array_ifact.copy())//n_average_fc).tolist())
        else:
            values_ifact.append(tb.array_ifact.copy())

    df = pd.DataFrame(list(zip(*values_iidx)), columns=columns)
    df.to_excel(excel_writer=path_values + "values_iidx_" + str(size) + ".xlsx")
    df = pd.DataFrame(list(zip(*values_ifact)), columns=columns)
    df.to_excel(excel_writer=path_values + "values_ifact_" + str(size) + ".xlsx")


def calculate_samples(modes, angles, size, normalize=0, create_table = False):
    values_ref = []
    values_id = []
    columns = []
    list_of_samples_map = []
    lowest_id_value = mh.inf
    highest_id_value = -mh.inf
    for i, j in zip(modes, angles):
        tb = TransformBlock(size, size, i, j, 0, size * 2 + 2, size * 2 + 2, 0)
        tb.calculate_pred_values()
        columns.append(i)
        values_ref.append(tb)
        list_of_samples_map.append(tb.ref)
        values_id.append(tb.ref_id)
        if lowest_id_value > tb.ref_id[0]:
            lowest_id_value = tb.ref_id[0]
        if highest_id_value < tb.ref_id[-1]:
            highest_id_value = tb.ref_id[-1]

    values_ref_array = []
    for i in values_ref:
        if normalize:
            i.normalize_ref()

        values_ref_array.append(i.transform_dict_to_array(lowest_id_value, highest_id_value, normalize))

    if create_table:

        rows = list(range(lowest_id_value, highest_id_value + 1))

        df = pd.DataFrame(list(zip(*values_ref_array)), index=rows, columns=columns)
        if normalize:
            df.to_excel(excel_writer=path_samples + "ref_" + str(size) + "_normalized" + ".xlsx")
        else:
            df.to_excel(excel_writer=path_samples + "ref_" + str(size) + ".xlsx")

    return list_of_samples_map


def map_index_to_constant(tb, coefficients, equations, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map):
    equations_constants = []
    equations_constants_samples = []
    equations_constants_reuse = []
    reused_equations = 0
    columns = []
    line_index = 0
    for line in equations:
        columns.append(line_index)
        line_constants = []
        line_constants_samples = []
        line_constants_reuse = []
        column_index = 0
        for equation in line:
            equation_constants = ""
            equation_constants_samples = ""
            for k in range(4):
                p, index, ref = re.findall(r'-?\d+', equation.split('+')[
                    k])  # get p[index] and ref from string containing and put it in two separately variables
                if p + '[' + index + ']' not in ft_coefficients[coefficients]:
                    p, index = simmetry_rule(p, index,
                                             coefficients)  # transform in a value that exists in the coefficients by the simmetry rule

                value = ft_coefficients[coefficients][p + '[' + index + ']']
                equation_constants += '(' + str(value) + ')*' + "ref[" + str(ref) + '] + '
                equation_constants_samples += '(' + str(value) + ')*' + str(tb.ref[int(ref)]) + ' + '

            equation_constants = equation_constants[:-3]
            equation_constants_samples = equation_constants_samples[:-3]
            #print(equation_constants_samples)
            if equation_constants_samples in equations_constants_samples_set:
                reused_equations += 1
                line_constants_reuse.append("REUSED: (" + equations_constants_reuse_map[equation_constants_samples] + ")")
            else:
                line_constants_reuse.append(equation_constants_samples)
                equations_constants_reuse_map[equation_constants_samples] = str(tb.predModeIntra) + ": " + str(line_index) + ", " + str(column_index)

            equations_constants_set.add(equation_constants)
            equations_constants_samples_set.add(equation_constants_samples)
            line_constants.append(equation_constants)
            line_constants_samples.append(equation_constants_samples)

            column_index += 1

        equations_constants.append(line_constants)
        equations_constants_samples.append(line_constants_samples)
        equations_constants_reuse.append(line_constants_reuse)
        line_index += 1

    return columns, equations_constants, equations_constants_samples, equations_constants_reuse, reused_equations, equations_constants_reuse_map


def calculate_equations(mode, angle, nTbW, nTbH, coefficients, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map, index_x = 0, index_y = 0, subset_size_x = 0, subset_size_y = 0, refidx = 0, cidx = 0,samples = False, reuse = False, create_table = False):
    if nTbW == nTbH:
        tb = TransformBlock(nTbW, nTbH, mode, angle, refidx, nTbW * 2 + 2, nTbH * 2 + 2, cidx)
    else:
        tb = TransformBlock(nTbW, nTbH, mode, angle, refidx, nTbW + nTbH + 1, nTbW + nTbH + 1, cidx)

    tb.calculate_pred_values()
    tb.calculate_equations_mode(create_table)
    equations = tb.get_equations(index_x, index_y, subset_size_x, subset_size_y)

    columns, equations_constants, equations_constants_samples, equations_constants_reuse, reused_equations, equations_constants_reuse_map = map_index_to_constant(tb, coefficients, equations, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map)
    #for equation in equations_constants_samples:
        #print(equation)

    equations_list = equations_constants
    if samples:
        equations_list = equations_constants_samples

    reuse_string = ""
    if reuse:
        reuse_string += "reuse_"
        equations_list = equations_constants_reuse

    if create_table:
        df = pd.DataFrame(list(zip(*equations_list)), columns=columns)
        excel_writer = pd.ExcelWriter(
            path_equations + "equations_" + reuse_string + coefficients + "_" + str(mode) + "_" + str(nTbW) + "x" + str(
                nTbH) + ".xlsx", engine='xlsxwriter')
        df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

        # Auto-adjust columns' width
        for column in df:
            column_width = 70
            col_iidx = df.columns.get_loc(column)
            excel_writer.sheets['equations'].set_column(col_iidx, col_iidx, column_width)

        excel_writer._save()

    '''for equation in equations_constants_set:
        print(equation)
    print(len(equations_constants_set))
    print(reused_equations)'''

    return equations_list, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map


def get_reference_number(equation):
    x = re.search("ref\\[-*[0-9]*", equation)
    y = re.findall("-*[0-9]+", x.group())
    return int(y[0])

def find_superior_ref_equation(equations_set):
    greater_value = -mh.inf
    greater_equation = None
    for equation in equations_set:
        value = get_reference_number(equation)
        if value > greater_value:
            greater_value = value
            greater_equation = equation

    return greater_equation

def generate_sorted_equations_set(equations_set, print_set):
    sorted_set = []
    while not equations_set == set():
        greater_equation = find_superior_ref_equation(equations_set)
        equations_set.remove(greater_equation)
        sorted_set.append(greater_equation)

    if print_set:
        i = 0
        for equation in sorted_set:
            i += 1
            print(i, equation)

    return sorted_set


def transform_coefficients(n_average_fc, n_average_fg, print_table, print_values_c):
    for column in range(0, 4):
        for line in range(0, 32, n_average_fc):
            average = 0
            for element in range(0, n_average_fc):
                coefficient = str(line + element) + '[' + str(column) + ']'
                average += get_coefficient_value("fc", coefficient)

            average = int(average / n_average_fc)

            for element in range(0, n_average_fc):
                fc_heuristic[str(line + element) + '[' + str(column) + ']'] = average

    for column in range(0, 4):
        for line in range(0, 32, n_average_fg):
            average = 0
            for element in range(0, n_average_fg):
                coefficient = str(line + element) + '[' + str(column) + ']'
                average += get_coefficient_value("fg", coefficient)

            average = int(average / n_average_fg)

            for element in range(0, n_average_fg):
                fg_heuristic[str(line + element) + '[' + str(column) + ']'] = average

    fc_column_list = [[] for _ in range(4)]
    fg_column_list = [[] for _ in range(4)]
    fc_column_list_normalized = [[] for _ in range(4)]
    fg_column_list_normalized = [[] for _ in range(4)]
    filter_column_list_normalized = [[] for _ in range(4)]

    for line in range(0, 32, n_average_fc):
        index_0 = str(line) + "[" + str(0) + "]"
        index_1 = str(line) + "[" + str(1) + "]"
        index_2 = str(line) + "[" + str(2) + "]"
        index_3 = str(line) + "[" + str(3) + "]"

        fc_column_list[0].append(fc_heuristic[index_0])
        if fc_heuristic[index_0] not in fc_column_list_normalized[0] and fc_heuristic[index_0] != 0:
            fc_column_list_normalized[0].append(fc_heuristic[index_0])
        if fc_heuristic[index_0] not in  filter_column_list_normalized[0] and fc_heuristic[index_0] != 0:
            filter_column_list_normalized[0].append(fc_heuristic[index_0])

        fc_column_list[1].append(fc_heuristic[index_1])
        if fc_heuristic[index_1] not in fc_column_list_normalized[1] and fc_heuristic[index_1] != 0:
            fc_column_list_normalized[1].append(fc_heuristic[index_1])
        if fc_heuristic[index_1] not in  filter_column_list_normalized[1] and fc_heuristic[index_1] != 0:
            filter_column_list_normalized[1].append(fc_heuristic[index_1])

        fc_column_list[2].append(fc_heuristic[index_2])
        if fc_heuristic[index_2] not in fc_column_list_normalized[2] and fc_heuristic[index_2] != 0:
            fc_column_list_normalized[2].append(fc_heuristic[index_2])
        if fc_heuristic[index_2] not in filter_column_list_normalized[2] and fc_heuristic[index_2] != 0:
            filter_column_list_normalized[2].append(fc_heuristic[index_2])

        fc_column_list[3].append(fc_heuristic[index_3])
        if fc_heuristic[index_3] not in fc_column_list_normalized[3] and fc_heuristic[index_3] != 0:
            fc_column_list_normalized[3].append(fc_heuristic[index_3])
        if fc_heuristic[index_3] not in  filter_column_list_normalized[3] and fc_heuristic[index_3] != 0:
            filter_column_list_normalized[3].append(fc_heuristic[index_3])

    for line in range(0, 32, n_average_fg):
        index_0 = str(line) + "[" + str(0) + "]"
        index_1 = str(line) + "[" + str(1) + "]"
        index_2 = str(line) + "[" + str(2) + "]"
        index_3 = str(line) + "[" + str(3) + "]"

        fg_column_list[0].append(fg_heuristic[index_0])
        if fg_heuristic[index_0] not in fg_column_list_normalized[0] and fg_heuristic[index_0] != 0:
            fg_column_list_normalized[0].append(fg_heuristic[index_0])
        if fg_heuristic[index_0] not in  filter_column_list_normalized[0] and fg_heuristic[index_0] != 0:
            filter_column_list_normalized[0].append(fg_heuristic[index_0])

        fg_column_list[1].append(fg_heuristic[index_1])
        if fg_heuristic[index_1] not in fg_column_list_normalized[1] and fg_heuristic[index_1] != 0:
            fg_column_list_normalized[1].append(fg_heuristic[index_1])
        if fg_heuristic[index_1] not in  filter_column_list_normalized[1] and fg_heuristic[index_1] != 0:
            filter_column_list_normalized[1].append(fg_heuristic[index_1])

        fg_column_list[2].append(fg_heuristic[index_2])
        if fg_heuristic[index_2] not in fg_column_list_normalized[2] and fg_heuristic[index_2] != 0:
            fg_column_list_normalized[2].append(fg_heuristic[index_2])
        if fg_heuristic[index_2] not in  filter_column_list_normalized[2] and fg_heuristic[index_2] != 0:
            filter_column_list_normalized[2].append(fg_heuristic[index_2])

        fg_column_list[3].append(fg_heuristic[index_3])
        if fg_heuristic[index_3] not in fg_column_list_normalized[3] and fg_heuristic[index_3] != 0:
            fg_column_list_normalized[3].append(fg_heuristic[index_3])
        if fg_heuristic[index_3] not in  filter_column_list_normalized[3] and fg_heuristic[index_3] != 0:
            filter_column_list_normalized[3].append(fg_heuristic[index_3])

    filter_coefficients = []
    filter_coefficients_normalized = []
    for line in range(0, 32, n_average_fc):
        index_0 = str(line) + "[" + str(0) + "]"
        index_1 = str(line) + "[" + str(1) + "]"
        index_2 = str(line) + "[" + str(2) + "]"
        index_3 = str(line) + "[" + str(3) + "]"

        filter_coefficients.append(fc_heuristic[index_0])
        filter_coefficients.append(fc_heuristic[index_1])
        filter_coefficients.append(fc_heuristic[index_2])
        filter_coefficients.append(fc_heuristic[index_3])

        if (fc_heuristic[index_0] not in filter_coefficients_normalized) and fc_heuristic[index_0] != 0:
            filter_coefficients_normalized.append(fc_heuristic[index_0])
        if (fc_heuristic[index_1] not in filter_coefficients_normalized) and fc_heuristic[index_1] != 0:
            filter_coefficients_normalized.append(fc_heuristic[index_1])
        if (fc_heuristic[index_2] not in filter_coefficients_normalized) and fc_heuristic[index_2] != 0:
            filter_coefficients_normalized.append(fc_heuristic[index_2])
        if (fc_heuristic[index_3] not in filter_coefficients_normalized) and fc_heuristic[index_3] != 0:
            filter_coefficients_normalized.append(fc_heuristic[index_3])

    for line in range(0, 32, n_average_fg):
        index_0 = str(line) + "[" + str(0) + "]"
        index_1 = str(line) + "[" + str(1) + "]"
        index_2 = str(line) + "[" + str(2) + "]"
        index_3 = str(line) + "[" + str(3) + "]"

        filter_coefficients.append(fg_heuristic[index_0])
        filter_coefficients.append(fg_heuristic[index_1])
        filter_coefficients.append(fg_heuristic[index_2])
        filter_coefficients.append(fg_heuristic[index_3])

        if (fg_heuristic[index_0] not in filter_coefficients_normalized) and fg_heuristic[index_0] != 0:
            filter_coefficients_normalized.append(fg_heuristic[index_0])
        if (fg_heuristic[index_1] not in filter_coefficients_normalized) and fg_heuristic[index_1] != 0:
            filter_coefficients_normalized.append(fg_heuristic[index_1])
        if (fg_heuristic[index_2] not in filter_coefficients_normalized) and fg_heuristic[index_2] != 0:
            filter_coefficients_normalized.append(fg_heuristic[index_2])
        if (fg_heuristic[index_3] not in filter_coefficients_normalized) and fg_heuristic[index_3] != 0:
            filter_coefficients_normalized.append(fg_heuristic[index_3])


    filter_column_list = []
    for fc_column, fg_column in zip(fc_column_list, fg_column_list):
        filter_column_list.append(fc_column + fg_column)

    if print_table:
        print("#####################fC################")
        #fc_column_sets = [set(), set(), set(), set()]
        for line in range(0, 32):
            index_0 = str(line) + "[" + str(0) + "]"
            index_1 = str(line) + "[" + str(1) + "]"
            index_2 = str(line) + "[" + str(2) + "]"
            index_3 = str(line) + "[" + str(3) + "]"
            '''fc_column_sets[0].add(fc_heuristic[index_0])
            fc_column_sets[1].add(fc_heuristic[index_1])
            fc_column_sets[2].add(fc_heuristic[index_2])
            fc_column_sets[3].add(fc_heuristic[index_3])'''
            print(index_0, fc_heuristic[index_0], get_coefficient_value("fc", index_0), index_1, fc_heuristic[index_1],
                  get_coefficient_value("fc", index_1), index_2, fc_heuristic[index_2],
                  get_coefficient_value("fc", index_2), index_3, fc_heuristic[index_3],
                  get_coefficient_value("fc", index_3))

        #print(fc_column_sets)
        print(fc_column_list)
        print(set(fc_coefficients.values()), len(set(fc_coefficients.values())))
        print(set(fc_heuristic.values()), len(set(fc_heuristic.values())))
        print("#####################fG################")
        #fg_column_sets = [set(),set(),set(),set()]
        for line in range(0, 32):
            index_0 = str(line) + "[" + str(0) + "]"
            index_1 = str(line) + "[" + str(1) + "]"
            index_2 = str(line) + "[" + str(2) + "]"
            index_3 = str(line) + "[" + str(3) + "]"
            '''fg_column_sets[0].add(fg_heuristic[index_0])
            fg_column_sets[1].add(fg_heuristic[index_1])
            fg_column_sets[2].add(fg_heuristic[index_2])
            fg_column_sets[3].add(fg_heuristic[index_3])'''
            print(index_0, fg_heuristic[index_0], get_coefficient_value("fg", index_0), index_1, fg_heuristic[index_1],
                  get_coefficient_value("fg", index_1), index_2, fg_heuristic[index_2],
                  get_coefficient_value("fg", index_2), index_3, fg_heuristic[index_3],
                  get_coefficient_value("fg", index_3))


        print("################ fC Column List ################")
        print(fc_column_list, len(fc_column_list[0]), len(fc_column_list[1]), len(fc_column_list[2]), len(fc_column_list[3]))
        print(fc_column_list_normalized, len(fc_column_list_normalized[0]), len(fc_column_list_normalized[1]), len(fc_column_list_normalized[2]) ,len(fc_column_list_normalized[3]))

        print("################ fG Column List ################")
        print(fg_column_list, len(fg_column_list[0]), len(fg_column_list[1]), len(fg_column_list[2]), len(fg_column_list[3]))
        print(fg_column_list_normalized, len(fg_column_list_normalized[0]), len(fg_column_list_normalized[1]), len(fg_column_list_normalized[2]) ,len(fg_column_list_normalized[3]))

        print("################ (fC + fG) List ################")
        print(filter_column_list, len(filter_column_list[0]), len(filter_column_list[1]), len(filter_column_list[2]), len(filter_column_list[3]))
        print(filter_column_list_normalized, len(filter_column_list_normalized[0]), len(filter_column_list_normalized[1]), len(filter_column_list_normalized[2]), len(filter_column_list_normalized[3]))

        print("################ (fC + fG) List ################")
        print(filter_coefficients, len(filter_coefficients))
        print(filter_coefficients_normalized, len(filter_coefficients_normalized))

        print("################ (fC + fG) Set ################")
        print(set(fg_heuristic.values()).union(set(fc_heuristic.values())),len(set(fg_heuristic.values()).union(set(fc_heuristic.values()))))

    if print_values_c:
        print("#####################fC################")
        i = 0
        for line in range(0, 32):
            i += 1
            index_0 = str(line) + "[" + str(0) + "]"
            index_1 = str(line) + "[" + str(1) + "]"
            index_2 = str(line) + "[" + str(2) + "]"
            index_3 = str(line) + "[" + str(3) + "]"
            print("{", str(fc_heuristic[index_0]) + ", " + str(fc_heuristic[index_1]) + ", " + str(
                fc_heuristic[index_2]) + ", " + str(fc_heuristic[index_3]), "},")

        print("#####################fG################")
        i = 0
        for line in range(0, 32):
            i += 1
            index_0 = str(line) + "[" + str(0) + "]"
            index_1 = str(line) + "[" + str(1) + "]"
            index_2 = str(line) + "[" + str(2) + "]"
            index_3 = str(line) + "[" + str(3) + "]"
            print("{", str(fg_heuristic[index_0]) + ", " + str(fg_heuristic[index_1]) + ", " + str(
                fg_heuristic[index_2]) + ", " + str(fg_heuristic[index_3]), "},")

    return filter_column_list, filter_column_list_normalized, filter_coefficients, filter_coefficients_normalized, set(fg_heuristic.values()).union(set(fc_heuristic.values()))


def append_list_without_repeat(l1, l2, repeat):
    l3 = l1[:]
    if repeat:
        [l3.append(i) for i in l2]
    else:
        [l3.append(i) for i in l2 if i not in l1]

    return l3

def generate_coefficients_for_parallel_prediction(filter_column_list, n, repeat):
    l_mcm_2 = append_list_without_repeat(filter_column_list[0], filter_column_list[1], repeat)
    l_mcm_3 = append_list_without_repeat(l_mcm_2, filter_column_list[2], repeat)
    l_mcm_4 = append_list_without_repeat(l_mcm_3, filter_column_list[3], repeat)
    l_mcm_minus_2 = append_list_without_repeat(filter_column_list[2], filter_column_list[3], repeat)
    l_mcm_minus_3 = append_list_without_repeat(filter_column_list[1], l_mcm_minus_2, repeat)

    coefficients_list = [filter_column_list[0], l_mcm_2, l_mcm_3]

    for i in range(n - 3):
        coefficients_list.append(l_mcm_4)

    coefficients_list.append(l_mcm_minus_3)
    coefficients_list.append(l_mcm_minus_2)
    coefficients_list.append(filter_column_list[3])

    return coefficients_list


def generate_mcm_blocks(filter_column_list_normalized):
    mcm_id = 0
    components = ""
    previous_column = []
    for column in filter_column_list_normalized:
        if not previous_column == column:
            previous_column = column.copy()
            components += "COMPONENT MCM_" + str(mcm_id) + "\n\tPORT (\n\t\tX : in std_logic_vector ( 7 downto 0 );"
            y_id = 1
            for coefficient in column[:-1]:
                components += "\n\t\tY" + str(y_id) + ", -- Y" + str(y_id) + " = ref[" + str(mcm_id) + "]*" + str(coefficient)
                y_id += 1

            components += "\n\t\tY" + str(y_id) + " : out std_logic_vector ( 15 downto 0 ) -- Y" + str(y_id) + " = ref[" + str(mcm_id) + "]*" + str(column[-1]) + "\n\t);\nEND COMPONENT;\n\n" #last one

            mcm_id += 1

    print(components)

def generate_port_mapping(filter_column_list_normalized, n):
    input_map = [ {} for _ in range(n + 3)]
    input_index = 0
    port_mapping = "BEGIN"
    m_index = 0
    m_counter = 0
    for filter_column, input_column  in zip(filter_column_list_normalized, input_map):
        port_mapping += "\n\tm" + str(m_counter) + " : MCM_" + str(m_index)
        port_mapping += "\n\tPORT MAP ( X => ref(" + str(m_counter) + ")"
        y_id = 1
        for coefficient in filter_column:
            port_mapping += ", Y" + str(y_id) + " => mcm_output(" + str(input_index) + ")"
            input_column[coefficient] = input_index
            input_index += 1
            y_id += 1
        port_mapping += " );"
        m_counter += 1
        if not 3 < m_counter < n:
            m_index += 1

    port_mapping += "\n"
    for i in range(n):
        port_mapping += "\teq" + str(i) + ": equation_block\n"
        port_mapping += "\tPORT MAP ( input_0 => eq_input(" + str(i) + ")(0), input_1 => eq_input(" + str(i) + ")(1), input_2 => eq_input("  + str(i) + ")(2), input_3 => eq_input(" + str(i) + ")(3), output => pred("  + str(i) + ") );\n"

    port_mapping += "\n"
    print(port_mapping)

    return input_map


def generate_mux(filter_column_list, input_map):
    mux_fg_fc = "case control is\n"
    size = len(filter_column_list[0])
    control_size = int(mh.log2(size))
    for j in range(len(filter_column_list[0])):
        line = np.array(filter_column_list)[:, j]
        mux_fg_fc += "\twhen " + '"' + str(bin(j)[2:].zfill(control_size)) + '"' + "=>\n"
        for i,coefficient in zip(range(0,4),line):
            if coefficient == 0:
                mux_fg_fc += "\t\teq_input(" + str(i) + ") <= " + '"' + "0000000000000000" + '"' + "; -- input " + str(i) + " <= 0 * ref[" + str(i) + "]\n"
            else:
                mux_fg_fc += "\t\teq_input(" + str(i) + ") <= input(" + str(input_map[i][coefficient]) + "); -- input " + str(i) + " <= " + str(coefficient) + " * ref[" + str(i) + "]\n"

    mux_fg_fc += "\twhen others => -- default case for not using latch\n\t\teq_input(0) <= " + '"' + "0000000000000000" + '"' + ";\n\t\teq_input(1) <= " + '"' + "0000000000000000" + '"' + ";\n\t\teq_input(2) <= " + '"' + "0000000000000000" + '"' + ";\n\t\teq_input(3) <= " + '"' + "0000000000000000" + '"' + ";\n"

    print(mux_fg_fc)

def generate_mux_n_samples(filter_column_list, input_map, n):
    mux_fg_fc = ""
    size = len(filter_column_list[0])
    for k in range(size):
        for i in range(n - 3):
            mux_fg_fc += "\n\t-- Eq " + str(i) + " Line " + str(k) + "\n"
            for j in range(4):
                mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(j) + ") <= mcm_output(" + str(input_map[i + j][filter_column_list[i + j][k + size*j]]) + "); -- input " + str(i) + "," + str(j) + " <= " + str(filter_column_list[i + j][k + size*j]) + " * ref[" + str(i + j) + "]\n"

        if n > 1:
            i = n - 3
            mux_fg_fc += "\n\t-- Eq " + str(i) + " Line " + str(k) + "\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(0) + ") <= mcm_output(" + str(input_map[i][filter_column_list[i][k]]) + "); -- input " + str(i) + ",0 <= " + str(filter_column_list[i][k]) + " * ref[" + str(i) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(1) + ") <= mcm_output(" + str(input_map[i + 1][filter_column_list[i + 1][k + size]]) + "); -- input " + str(i) + ",1 <= " + str(filter_column_list[i + 1][k + size]) + " * ref[" + str(i + 1) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(2) + ") <= mcm_output(" + str(input_map[i + 2][filter_column_list[i + 2][k + size*2]]) + "); -- input " + str(i) + ",2 <= " + str(filter_column_list[i + 2][k + size*2]) + " * ref[" + str(i + 2) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(3) + ") <= mcm_output(" + str(input_map[i + 3][filter_column_list[i + 3][k + size*2]]) + "); -- input " + str(i) + ",3 <= " + str(filter_column_list[i + 3][k + size*2]) + " * ref[" + str(i + 3) + "]\n"

            i += 1
            mux_fg_fc += "\n\t-- Eq " + str(i) + " Line " + str(k) + "\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(0) + ") <= mcm_output(" + str(input_map[i][filter_column_list[i][k]]) + "); -- input " + str(i) + ",0 <= " + str(filter_column_list[i][k]) + " * ref[" + str(i) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(1) + ") <= mcm_output(" + str(input_map[i + 1][filter_column_list[i + 1][k + size]]) + "); -- input " + str(i) + ",1 <= " + str(filter_column_list[i + 1][k + size]) + " * ref[" + str(i + 1) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(2) + ") <= mcm_output(" + str(input_map[i + 2][filter_column_list[i + 2][k + size]]) + "); -- input " + str(i) + ",2 <= " + str(filter_column_list[i + 2][k + size]) + " * ref[" + str(i + 2) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(3) + ") <= mcm_output(" + str(input_map[i + 3][filter_column_list[i + 3][k + size]]) + "); -- input " + str(i) + ",3 <= " + str(filter_column_list[i + 3][k + size]) + " * ref[" + str(i + 3) + "]\n"

            i += 1
            mux_fg_fc += "\n\t-- Eq " + str(i) + " Line " + str(k) + "\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(0) + ") <= mcm_output(" + str(input_map[i][filter_column_list[i][k]]) + "); -- input " + str(i) + ",0 <= " + str(filter_column_list[i][k]) + " * ref[" + str(i) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(1) + ") <= mcm_output(" + str(input_map[i + 1][filter_column_list[i + 1][k]]) + "); -- input " + str(i) + ",1 <= " + str(filter_column_list[i + 1][k]) + " * ref[" + str(i + 1) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(2) + ") <= mcm_output(" + str(input_map[i + 2][filter_column_list[i + 2][k]]) + "); -- input " + str(i) + ",2 <= " + str(filter_column_list[i + 2][k]) + " * ref[" + str(i + 2) + "]\n"
            mux_fg_fc += "\tinput(" + str(k) + ")(" + str(i) + ")(" + str(3) + ") <= mcm_output(" + str(input_map[i + 3][filter_column_list[i + 3][k]]) + "); -- input " + str(i) + ",3 <= " + str(filter_column_list[i + 3][k]) + " * ref[" + str(i + 3) + "]\n"

    print(mux_fg_fc)


def generate_mux_n_samples_sc(filter_column_list, input_map, n):
    mux_fg_fc = "case control is\n"
    size = len(filter_column_list[0])
    control_size = int(mh.log2(size))
    for k in range(size):
        mux_fg_fc += "\n\twhen " + '"' + str(bin(k)[2:].zfill(control_size)) + '"' + "=>"
        for i in range(n - 3):
            mux_fg_fc += "\n\t\t-- Eq " + str(i) + "\n"
            for j in range(4):
                mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(j) + ") <= mcm_output(" + str(input_map[i + j][filter_column_list[i + j][k + size*j]]) + "); -- input " + str(i) + ",0 <= " + str(filter_column_list[i + j][k + size*j]) + " * ref[" + str(i + j) + "]\n"

        i = n - 3
        mux_fg_fc += "\n\t\t-- Eq " + str(i) + "\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(0) + ") <= mcm_output(" + str(input_map[i][filter_column_list[i][k]]) + "); -- input " + str(i) + ",0 <= " + str(filter_column_list[i][k]) + " * ref[" + str(i) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(1) + ") <= mcm_output(" + str(input_map[i + 1][filter_column_list[i + 1][k + size]]) + "); -- input " + str(i) + ",1 <= " + str(filter_column_list[i + 1][k + size]) + " * ref[" + str(i + 1) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(2) + ") <= mcm_output(" + str(input_map[i + 2][filter_column_list[i + 2][k + size*2]]) + "); -- input " + str(i) + ",2 <= " + str(filter_column_list[i + 2][k + size*2]) + " * ref[" + str(i + 2) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(3) + ") <= mcm_output(" + str(input_map[i + 3][filter_column_list[i + 3][k + size*2]]) + "); -- input " + str(i) + ",3 <= " + str(filter_column_list[i + 3][k + size*2]) + " * ref[" + str(i + 3) + "]\n"

        i += 1
        mux_fg_fc += "\n\t\t-- Eq " + str(i) + "\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(0) + ") <= mcm_output(" + str(input_map[i][filter_column_list[i][k]]) + "); -- input " + str(i) + ",0 <= " + str(filter_column_list[i][k]) + " * ref[" + str(i) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(1) + ") <= mcm_output(" + str(input_map[i + 1][filter_column_list[i + 1][k + size]]) + "); -- input " + str(i) + ",1 <= " + str(filter_column_list[i + 1][k + size]) + " * ref[" + str(i + 1) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(2) + ") <= mcm_output(" + str(input_map[i + 2][filter_column_list[i + 2][k + size]]) + "); -- input " + str(i) + ",2 <= " + str(filter_column_list[i + 2][k + size]) + " * ref[" + str(i + 2) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(3) + ") <= mcm_output(" + str(input_map[i + 3][filter_column_list[i + 3][k + size]]) + "); -- input " + str(i) + ",3 <= " + str(filter_column_list[i + 3][k + size]) + " * ref[" + str(i + 3) + "]\n"

        i += 1
        mux_fg_fc += "\n\t\t-- Eq " + str(i) + "\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(0) + ") <= mcm_output(" + str(input_map[i][filter_column_list[i][k]]) + "); -- input " + str(i) + ",0 <= " + str(filter_column_list[i][k]) + " * ref[" + str(i) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(1) + ") <= mcm_output(" + str(input_map[i + 1][filter_column_list[i + 1][k]]) + "); -- input " + str(i) + ",1 <= " + str(filter_column_list[i + 1][k]) + " * ref[" + str(i + 1) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(2) + ") <= mcm_output(" + str(input_map[i + 2][filter_column_list[i + 2][k]]) + "); -- input " + str(i) + ",2 <= " + str(filter_column_list[i + 2][k]) + " * ref[" + str(i + 2) + "]\n"
        mux_fg_fc += "\t\teq_input(" + str(i) + ")(" + str(3) + ") <= mcm_output(" + str(input_map[i + 3][filter_column_list[i + 3][k]]) + "); -- input " + str(i) + ",3 <= " + str(filter_column_list[i + 3][k]) + " * ref[" + str(i + 3) + "]\n"

    mux_fg_fc += "\twhen others => -- default case for not using latch\n\t\teq_input(0) <= " + '"' + "0000000000000000" + '"' + ";\n\t\teq_input(1) <= " + '"' + "0000000000000000" + '"' + ";\n\t\teq_input(2) <= " + '"' + "0000000000000000" + '"' + ";\n\t\teq_input(3) <= " + '"' + "0000000000000000" + '"' + ";\n"

    print(mux_fg_fc)


def generate_rom(filter_column_lists, filter_coefficients_normalized):
    data_map = {}
    size = 256
    control_size = int(mh.log2(size))
    index = 0
    rom = "\t" + str(index) + ' => "' + str(bin(index)[2:].zfill(control_size)) + '",\n'
    for coefficient in filter_coefficients_normalized:
        data_map[coefficient] = index
        index += 1
        rom += "\t" + str(index) + ' => "' + str(bin(coefficient)[2:].zfill(control_size)) + '",\n'


    i = 0
    size = 64
    control_size = int(mh.log2(size))
    for coefficient_1, coefficient_2, coefficient_3, coefficient_4 in zip(filter_column_lists[0],filter_column_lists[1],filter_column_lists[2],filter_column_lists[3]):
        rom += "\twhen " + '"' + str(bin(i)[2:].zfill(control_size)) + '"' + " => data_1 <= coefficients(" + str(data_map[coefficient_1]) + "); data_2 <= coefficients(" + str(data_map[coefficient_2]) + "); data_3 <= coefficients(" + str(data_map[coefficient_3]) + "); data_4 <= coefficients(" + str(data_map[coefficient_4]) + ");\n"
        i += 1

    print(rom)


def generate_angular_mode_mapping(f, state_mapping, size_x, size_y, iteration_size, block_size_x, block_size_y, iteration_only = False):
    #print(state_mapping.values())
    for equations, states in zip(state_mapping.keys(), state_mapping.values()):
        output_index = 0
        mode_index = 0
        if_string = "\nelsif control ="
        #print(states) 
        for state in states:
            if not iteration_only:
                if_string += " " + '"' + str(bin(state[0])[2:].zfill(mh.ceil(mh.log2(size_x)))) + str(
                    bin(state[1])[2:].zfill(mh.ceil(mh.log2(size_y)))) + str(
                    bin(state[2])[2:].zfill(mh.ceil(mh.log2(iteration_size)))) + '"' + " or control ="
            else:
                if_string += " " + '"' + str(bin(state[2])[2:].zfill(mh.ceil(mh.log2(iteration_size)))) + '"' + " or control ="
        if_string = if_string[:-12]
        if_string += "then\n"
        f.write(if_string)
        for equation in equations:    
            f.write("\toutput(" + str(mode_index) + ", " + str(output_index) + ") <= input(" + str(equation) + ");\n")
            output_index += 1
            if output_index ==  block_size_x*block_size_y:
                output_index = 0
                mode_index += 1


def angular_input_mapping(modes, angles, parallel_modes_list, nTbW, nTbH, initial_index_x, initial_index_y,
                            final_index_x, final_index_y, subset_size_x, subset_size_y, refidx, samples_on, reuse_on, coefficients_table):
    size = len(parallel_modes_list)
    iterations = int(len(parallel_modes_list))
    state_mapping = {}
    f = open("input_" + str(subset_size_x) + "x" + str(subset_size_y) + "_" + str(nTbW) + "x" + str(nTbH) + ".txt",
             "w")
    c = open("control_" + str(subset_size_x) + "x" + str(subset_size_y) + "_" + str(nTbW) + "x" + str(nTbH) + ".txt",
             "w")
    block_counter = 0
    for index_x in range(initial_index_x, final_index_x, subset_size_x):
        for index_y in range(initial_index_y, final_index_y, subset_size_y):
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
                for mode, angle in zip(modes_subset, angles_subset):
                    mode_exit_mapping = []
                    equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = calculate_equations(
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
                        subset_size_x=subset_size_x,
                        subset_size_y=subset_size_y,
                        refidx=refidx,
                        samples=samples_on,
                        reuse=reuse_on,
                        create_table=False)

                    for equations_column in equations:
                        for equation in equations_column:
                            if equation not in unit_equation_mapping.keys():
                                unit_equation_mapping[equation] = unit_index
                                unit_index += 1

                            mode_exit_mapping.append(unit_equation_mapping[equation])


                unit_equation_mapping, control_mapping = transform_pixel_to_sample_array(unit_equation_mapping)
                control_sequence = generate_control_sequence(control_mapping, coefficients_table)
                generate_control_sequence_file(c, control_sequence, "fc", (int(index_x/subset_size_x), int(index_y/subset_size_y), i) ,int(final_index_x / subset_size_x),
                                      int(final_index_y / subset_size_y), len(parallel_modes_list))

                if tuple(unit_equation_mapping) not in state_mapping.keys():
                    state_mapping[tuple(unit_equation_mapping)] = []
                state_mapping[tuple(unit_equation_mapping)].append(
                    (int(index_x / subset_size_x), int(index_y / subset_size_y), i))
                index += parallel_modes_number

            block_counter += 1

    #print(state_mapping)
    generate_angular_input_mapping(f, state_mapping, int(final_index_x / subset_size_x),
                                      int(final_index_y / subset_size_y), len(parallel_modes_list), subset_size_x,
                                      subset_size_y)

    f.close()
    c.close()

def generate_angular_input_mapping(f, state_mapping, size_x, size_y, iteration_size, block_size_x, block_size_y, iteration_only = False):
    #print(state_mapping.values())
    for equations, states in zip(state_mapping.keys(), state_mapping.values()):
        ref_index = 0
        input_index = 0
        if_string = "\nelsif control ="
        #print(states)
        for state in states:
            if not iteration_only:
                if_string += " " + '"' + str(bin(state[0])[2:].zfill(mh.ceil(mh.log2(size_x)))) + str(
                    bin(state[1])[2:].zfill(mh.ceil(mh.log2(size_y)))) + str(
                    bin(state[2])[2:].zfill(mh.ceil(mh.log2(iteration_size)))) + '"' + " or control ="
            else:
                if_string += " " + '"' + str(bin(state[2])[2:].zfill(mh.ceil(mh.log2(iteration_size)))) + '"' + " or control ="

        if_string = if_string[:-12]
        if_string += "then\n"
        f.write(if_string)
        for equation in equations:
            f.write("\tinput(" + str(input_index) + ")(" + str(ref_index) + ") <= " + str(equation) + ";\n")
            ref_index += 1
            if ref_index ==  4:
                ref_index = 0
                input_index += 1

def transform_pixel_to_sample_array(unit_equation_mapping):
    top_samples = []
    left_samples = []
    new_unit_equation_mapping = []
    constants_mapping = []
    for equation in unit_equation_mapping:
        constants = []
        #print(equation)
        for k in range(4):
            constant, index_x, index_y = re.findall(r'-?\d+', equation.split('+')[
                k])  # get p[index] and ref from string containing and put it in two separately variables
            if int(index_x) == -1 and int(index_y) == -1:
                new_unit_equation_mapping.append("top_samples(" + str(0) + ")")
            elif int(index_x) == -1:
                new_unit_equation_mapping.append("left_samples(" + str(index_y) + ")")
            elif int(index_y) == -1:
                new_unit_equation_mapping.append("top_samples(" + str(int(index_x) + 1) + ")")
            else:
                throw_error("Undefined sample")
            constants.append(constant)

        constants_mapping.append(constants)

    return new_unit_equation_mapping, constants_mapping

def generate_control_sequence(control_mapping, coefficients):
    #print(control_mapping)
    control_sequence = []
    for state in control_mapping:
        for line_index in range(len(coefficients[0])):
            if str(state[0]) == str(coefficients[0][line_index]) and str(state[1]) == str(coefficients[1][line_index]) and str(state[2]) == str(coefficients[2][line_index]) and str(state[3]) == str(coefficients[3][line_index]):
                control_sequence.append(line_index)

    return control_sequence


def generate_control_sequence_file(c, control_sequence, coefficients_string, state, size_x, size_y, iteration_size, block = 0):
    """DANGER ONLY WORKING FOR N16"""
    if coefficients_string == "fc":
        coefficient_bit = 0
    else:
        coefficient_bit = 1

    c.write(
        "\toutput_control <= " + '"' + str(bin(state[0])[2:].zfill(mh.ceil(mh.log2(size_x)))) + str(
        bin(state[1])[2:].zfill(mh.ceil(mh.log2(size_y)))) + str(
        bin(state[2])[2:].zfill(mh.ceil(mh.log2(iteration_size)))) + '"' + ";\n")
    c.write(
        "\tinput_control <= " + '"' + str(bin(state[0])[2:].zfill(mh.ceil(mh.log2(size_x)))) + str(
            bin(state[1])[2:].zfill(mh.ceil(mh.log2(size_y)))) + str(
            bin(state[2])[2:].zfill(mh.ceil(mh.log2(iteration_size)))) + '"' + ";\n")
    for control_bits, unit_index in zip(control_sequence, range(len(control_sequence))):
        if block > 0:
            c.write("\tcontrol_block_" + str(block) + "(" + str(unit_index) + ") <= " + '"' + str(coefficient_bit) + str(
                control_bits) + '"' + ";\n")
        else:
            c.write("\tunit_control(" + str(unit_index) + ") <= " + '"' + str(coefficient_bit) + str(control_bits) + '"' + ";\n")

    c.write("\twait for 5 ns;\n")


def generate_samples_buffer(seed, samples_size_top, samples_size_left):
    rm.seed(seed)  # set seed so that input has the same values for all tests
    f = open("random_inputs.txt","w")
    top_samples = []
    left_samples = []
    for i in range(samples_size_top):
        input_value = rm.randint(0, 255)
        top_samples.append(input_value)
        f.write("top_samples(" + str(i) + ") <= std_logic_vector(to_unsigned(" + str(input_value) + ",8));\n")

    for i in range(samples_size_left):
        input_value = rm.randint(0, 255)
        left_samples.append(input_value)
        f.write("left_samples(" + str(i) + ") <= std_logic_vector(to_unsigned(" + str(input_value) + ",8));\n")

    f.close()
    return top_samples, left_samples


def generate_mapping_states(output_state_mapping, input_state_mapping, parallel_modes_list, modes, angles, nTbW, nTbH, index_x, index_y, subset_size_x, subset_size_y, final_index_x, final_index_y,iterations, coefficients_table):
    equations_constants_set = set()
    equations_constants_samples_set = set()
    equations_constants_reuse_map = {}
    control_list = []
    index = 0
    for i in range(iterations):
        unit_equation_mapping = {}
        unit_index = 0
        parallel_modes_number = parallel_modes_list[i]
        modes_subset = modes[index:index + parallel_modes_number]
        angles_subset = angles[index:index + parallel_modes_number]
        modes_states_list = []
        exit_buffer_unit_mapping = []
        for mode, angle in zip(modes_subset, angles_subset):
            mode_exit_mapping = []
            equations, equations_constants_reuse, equations_constants_set, equations_constants_samples_set, equations_constants_reuse_map = calculate_equations(
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
                subset_size_x=subset_size_x,
                subset_size_y=subset_size_y,
                refidx= 0,
                samples= True,
                reuse= False,
                create_table=False)

            for equations_column in equations:
                for equation in equations_column:
                    if equation not in unit_equation_mapping.keys():
                        unit_equation_mapping[equation] = unit_index
                        unit_index += 1

                    mode_exit_mapping.append(unit_equation_mapping[equation])
                    exit_buffer_unit_mapping.append(unit_equation_mapping[equation])

            modes_states_list.append(mode_exit_mapping)

        unit_equation_mapping, control_mapping = transform_pixel_to_sample_array(unit_equation_mapping)
        control_sequence = generate_control_sequence(control_mapping, coefficients_table)
        control_list.append(control_sequence)

        if tuple(exit_buffer_unit_mapping) not in output_state_mapping.keys():
            output_state_mapping[tuple(exit_buffer_unit_mapping)] = []
        output_state_mapping[tuple(exit_buffer_unit_mapping)].append(
            (int(index_x / subset_size_x), int(index_y / subset_size_y), i))

        if tuple(unit_equation_mapping) not in input_state_mapping.keys():
            input_state_mapping[tuple(unit_equation_mapping)] = []
        input_state_mapping[tuple(unit_equation_mapping)].append(
            (int(index_x / subset_size_x), int(index_y / subset_size_y), i))
        index += parallel_modes_number

    return output_state_mapping, input_state_mapping, control_list


#GERADO POR DEEPSEEK
def compare_files(file1_path, file2_path, output_path):
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            lines1 = file1.readlines()
            lines2 = file2.readlines()
    except FileNotFoundError as e:
        print(f"Erro ao abrir arquivos: {e}")
        return

    max_lines = max(len(lines1), len(lines2))
    output_lines = []
    version1_lines = []
    version2_lines = []
    in_difference_block = False

    for i in range(max_lines):
        line1 = lines1[i].rstrip() if i < len(lines1) else None
        line2 = lines2[i].rstrip() if i < len(lines2) else None

        if line1 == line2:
            # Se encontramos linha igual, finaliza qualquer bloco de diferença existente
            if in_difference_block:
                output_lines.append("#if VERSION1\n")
                output_lines.extend(version1_lines)
                output_lines.append("#else  // VERSION2\n")
                output_lines.extend(version2_lines)
                output_lines.append("#endif\n")
                version1_lines = []
                version2_lines = []
                in_difference_block = False
            output_lines.append(f"{line1}\n" if line1 is not None else "\n")
        else:
            # Começa novo bloco de diferença se não estivermos em um
            if not in_difference_block:
                in_difference_block = True

            # Armazena as linhas diferentes
            if line1 is not None:
                version1_lines.append(f"{line1}\n")
            if line2 is not None:
                version2_lines.append(f"{line2}\n")

    # Processa qualquer bloco de diferença pendente no final do arquivo
    if in_difference_block:
        output_lines.append("#if VERSION1\n")
        output_lines.extend(version1_lines)
        output_lines.append("#else  // VERSION2\n")
        output_lines.extend(version2_lines)
        output_lines.append("#endif\n")

    try:
        with open(output_path, 'w') as output_file:
            output_file.writelines(output_lines)
        print(f"Arquivo de saída gerado com sucesso: {output_path}")
    except IOError as e:
        print(f"Erro ao escrever arquivo de saída: {e}")