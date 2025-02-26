import pandas as pd
import math as mh
import re
from collections import defaultdict
from transform_block import TransformBlock

path = "./"
path_output_metrics = path + "output/metrics/"
path_output_blocks = path + "output/mcm_block/"
path_output_states = path + "output/states/"
path_values = path + "output/values_iidx_ifact/"
path_samples = path + "output/samples/"
path_planar = path + "output/planar/"
path_equations = "output/equations/"

modes1 = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
          61, 62, 63, 64, 65, 66]
modes2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34]
modes3 = [2, 3, 7, 10, 18, 23, 26, 30, 33, 34, 35, 43, 46, 49, 50, 54]
modes4 = [35]
modes5 = [35, 54]
all_modes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
             58, 59, 60, 61, 62, 63, 64, 65, 66]
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
angles_negative = [-32, -29, -26, -23, -20, -18, -16, -14, -12, -10, -8, -6, -4, -3, -2, -1, 0]

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


def calculate_iidx_ifact(modes, angles, size):
    values_ifact = []
    values_iidx = []
    columns = []
    for i, j in zip(modes, angles):
        tb = TransformBlock(size, size, i, j, 0, size * 2 + 2, size * 2 + 2, 0)
        tb.calculate_constants_mode()
        columns.append(i)
        values_iidx.append(tb.array_iidx.copy())
        values_ifact.append(tb.array_ifact.copy())

    df = pd.DataFrame(list(zip(*values_iidx)), columns=columns)
    df.to_excel(excel_writer=path_values + "values_iidx_" + str(size) + ".xlsx")
    df = pd.DataFrame(list(zip(*values_ifact)), columns=columns)
    df.to_excel(excel_writer=path_values + "values_ifact_" + str(size) + ".xlsx")


def calculate_samples(modes, angles, size, normalize=0):
    values_ref = []
    values_id = []
    columns = []
    lowest_id_value = mh.inf
    highest_id_value = -mh.inf
    for i, j in zip(modes, angles):
        tb = TransformBlock(size, size, i, j, 0, size * 2 + 2, size * 2 + 2, 0)
        tb.calculate_pred_values()
        columns.append(i)
        values_ref.append(tb)
        values_id.append(tb.ref_id)
        if (lowest_id_value > tb.ref_id[0]):
            lowest_id_value = tb.ref_id[0]
        if (highest_id_value < tb.ref_id[-1]):
            highest_id_value = tb.ref_id[-1]

    values_ref_array = []
    for i in values_ref:
        if (normalize):
            i.normalize_ref()

        values_ref_array.append(i.transform_dict_to_array(lowest_id_value, highest_id_value, normalize))

    rows = list(range(lowest_id_value, highest_id_value + 1))

    df = pd.DataFrame(list(zip(*values_ref_array)), index=rows, columns=columns)
    if (normalize):
        df.to_excel(excel_writer=path_samples + "ref_" + str(size) + "_normalized" + ".xlsx")
    else:
        df.to_excel(excel_writer=path_samples + "ref_" + str(size) + ".xlsx")


def calculate_equations(modes, angles, size, coefficients):
    for i, j in zip(modes, angles):
        tb = TransformBlock(size, size, i, j, 0, size * 2 + 2, size * 2 + 2, 0)
        equations = tb.calculate_equations_mode()
        equations_constants = []
        equations_constants_set = set()
        reused_equation = 0
        columns = []
        line_index = 0
        for line in equations:
            columns.append(line_index)
            line_index += 1
            line_constants = []
            for equation in line:
                equation_constants = ""
                for k in range(4):
                    p, index, ref = re.findall(r'-?\d+', equation.split('+')[
                        k])  # get p[index] and ref from string containing and put it in two separately variables
                    if (p + '[' + index + ']' not in ft_coefficients[coefficients]):
                        p, index = simmetry_rule(p, index,
                                                 coefficients)  # transform in a value that exists in the coefficients by the simmetry rule

                    value = ft_coefficients[coefficients][p + '[' + index + ']']
                    equation_constants += '(' + str(value) + ')*' + "ref[" + str(ref) + '] + '

                equation_constants = equation_constants[:-3]
                line_constants.append(equation_constants)

                if equation_constants in equations_constants_set:
                    reused_equation += 1

                equations_constants_set.add(equation_constants)

            equations_constants.append(line_constants)

        df = pd.DataFrame(list(zip(*equations_constants)), columns=columns)
        excel_writer = pd.ExcelWriter(
            path_equations + "equations_constants_" + coefficients + "_" + str(i) + "_" + str(size) + "x" + str(
                size) + ".xlsx", engine='xlsxwriter')
        df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

        # Auto-adjust columns' width
        for column in df:
            column_width = 70
            col_iidx = df.columns.get_loc(column)
            excel_writer.sheets['equations'].set_column(col_iidx, col_iidx, column_width)

        excel_writer._save()
        print(len(equations_constants_set))
        print(reused_equation)


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

    '''print("############## fC #################")
    for key in fc_heuristic:
        print("%s : %s" % (key, fc_heuristic[key]))'''

    for column in range(0, 4):
        for line in range(0, 32, n_average_fg):
            average = 0
            for element in range(0, n_average_fg):
                coefficient = str(line + element) + '[' + str(column) + ']'
                average += get_coefficient_value("fg", coefficient)

            average = int(average / n_average_fg)

            for element in range(0, n_average_fg):
                fg_heuristic[str(line + element) + '[' + str(column) + ']'] = average

    '''print("############## fG #################")
    for key in fg_heuristic:
        print("%s : %s" % (key, fg_heuristic[key]))'''

    if (print_table):
        print("#####################fC################")
        for line in range(0, 32):
            index_0 = str(line) + "[" + str(0) + "]"
            index_1 = str(line) + "[" + str(1) + "]"
            index_2 = str(line) + "[" + str(2) + "]"
            index_3 = str(line) + "[" + str(3) + "]"
            print(index_0, fc_heuristic[index_0], get_coefficient_value("fc", index_0), index_1, fc_heuristic[index_1],
                  get_coefficient_value("fc", index_1), index_2, fc_heuristic[index_2],
                  get_coefficient_value("fc", index_2), index_3, fc_heuristic[index_3],
                  get_coefficient_value("fc", index_3))

        print(set(fc_coefficients.values()), len(set(fc_coefficients.values())))
        print(set(fc_heuristic.values()), len(set(fc_heuristic.values())))
        print("#####################fG################")
        for line in range(0, 32):
            index_0 = str(line) + "[" + str(0) + "]"
            index_1 = str(line) + "[" + str(1) + "]"
            index_2 = str(line) + "[" + str(2) + "]"
            index_3 = str(line) + "[" + str(3) + "]"
            print(index_0, fg_heuristic[index_0], get_coefficient_value("fg", index_0), index_1, fg_heuristic[index_1],
                  get_coefficient_value("fg", index_1), index_2, fg_heuristic[index_2],
                  get_coefficient_value("fg", index_2), index_3, fg_heuristic[index_3],
                  get_coefficient_value("fg", index_3))

        print(set(fg_coefficients.values()), len(set(fg_coefficients.values())))
        print(set(fg_heuristic.values()), len(set(fg_heuristic.values())))

        print("################(fC + fG) Set################")
        print(set(fg_heuristic.values()).union(set(fc_heuristic.values())),
              len(set(fg_heuristic.values()).union(set(fc_heuristic.values()))))

    if (print_values_c):
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