import numpy as np
import math as mh
import re
import random as rm
import generator as gen
from numpy.f2py.auxfuncs import throw_error
path_tests = "./output/tests/"

def map_pixel_to_reference(input_pixels, mode_tb, lower_limit, upper_limit):
    ref = {}
    for key, value in zip(mode_tb.ref.keys(),mode_tb.ref.values()):
        if mode_tb.predModeIntra < 34:
            if key <= (lower_limit + 1):
                break
        else:
            if key >= (upper_limit + 1):
                break

        x,y = re.findall(r'-?\d+', value) #Get x and y value from string
        ref[key] = input_pixels[(int(x),int(y))]


def assert_equals_angular(p, mode, angle, block_size, lower_limit, upper_limit):
    test_counter = 0
    nTbH = block_size
    nTbW = block_size
    predSamples = {}

    mode_tb = tb.TransformBlock(block_size, block_size, mode, angle, 0, block_size * 2 + 2, block_size * 2 + 2, 0)
    mode_tb.calculate_pred_values()

    ref = map_pixel_to_reference(p, mode_tb, lower_limit, upper_limit)
    # Find if mode uses fC or fG coefficients
    nTbS = int(mh.log2(block_size) + mh.log2(block_size)) >> 1
    minDistVerHor = min(abs(mode - 50), abs(mode - 18))
    if (minDistVerHor > intraHorVerDistThres[nTbS]):
        filterFlag = 1
    else:
        filterFlag = 0

    '''for key, value in zip(ref.keys(),ref.values()):
        print(key, value)'''

    for x in range(nTbW):
        for y in range(nTbH):
            iIdx = ((y + 1) * angle) >> 5
            iFact = ((y + 1) * angle) & 31
            predSamples[(x, y)] = calculate_pred_y(ref, x, iIdx, iFact, filterFlag)

    test_counter = 0
    y = 0
    x = 0
    f = open(path_tests + "output_results_mode" + str(mode) + ".txt", "r")
    for line in f:
        if (predSamples[(x, y)] == int(line, 2)):
            test_counter += 1
        else:
            error = str(x) + ", " + str(y) + " : " + str(predSamples[(x, y)]) + " /= " + str(int(line, 2))
            print(error)

        y += 1
        if (y == 32):
            y = 0
            x += 1

    print("Mode " + str(mode) + " > Correct Values : " + str(test_counter) + "/" + str(32 * block_size))
    f.close()


def assert_equals_planar(p, x_input):
    nTbH = 32
    nTbW = 32
    predSamples = {}
    predV = {}
    predH = {}
    f = open(path_tests + "planar_out.txt", "w")
    for x in range(x_input):
        for y in range(nTbH):
            predV[(x, y)] = ((nTbH - 1 - y) * p[(x, -1)] + (y + 1) * p[(-1, nTbH)]) << int(mh.log2(nTbW))
            predH[(x, y)] = ((nTbW - 1 - x) * p[(-1, y)] + (x + 1) * p[(nTbW, -1)]) << int(mh.log2(nTbH))
            predSamples[(x, y)] = (predV[(x, y)] + predH[(x, y)] + nTbH * nTbW) >> int(
                mh.log2(nTbW) + mh.log2(nTbH) + 1)
            f.write(str(x) + ", " + str(y) + " : " + str(predSamples[(x, y)]) + "\n")

    test_counter = 0
    y = 0
    x = 0
    f = open(path_tests + "output_results_planar.txt", "r")
    for line in f:
        if (predSamples[(x, y)] == int(line, 2)):
            test_counter += 1
        else:
            error = str(x) + ", " + str(y) + " : " + str(predSamples[(x, y)]) + " /= " + str(int(line, 2))
            print(error)

        y += 1
        if (y == 32):
            y = 0
            x += 1

    print("Planar Correct Values : " + str(test_counter) + "/" + str(32 * x_input))
    f.close()


def assert_equals_dc(p, block_size, result):
    nTbH = block_size
    nTbW = block_size
    x_sum = 0
    for x in range(nTbW):
        x_sum += p[(x, -1)]

    y_sum = 0
    for y in range(nTbH):
        y_sum += p[(-1, y)]

    sum = (x_sum + y_sum + 32) >> 6

    if (result == sum):
        print("DC Test Passed")
    else:
        print("DC Test Failed: " + str(sum) + " != " + str(result) + "\n")


# Automated tests
def random_generate_input(f, mode, angle, base, size):
    input = {}
    for n in range(base, base + size):
        iIdx = ((n + 1) * angle) >> 5
        new_base = iIdx
        for i in range(0, 4):
            index = 0 + iIdx + i
            if (index not in input):
                input[index] = rm.randint(0, 255)
                f.write("ref_" + str(index) + " : " + str(input[index]) + "\n")

    return input, new_base


def generate_output(f, tb, angle, initial_index_x, initial_index_y, final_index_x, final_index_y, top_samples, left_samples, filter_table):
    pred_result = []
    for y in range(initial_index_y, final_index_y):
        for x in range(initial_index_x, final_index_x):
            iIdx = ((y + 1) * angle) >> 5
            iFact = ((y + 1) * angle) & 31
            pred_result.append(calculate_pred_y(tb, x, iIdx, iFact, top_samples, left_samples, filter_table))
            # print(str(calculate_pred_y(input,x_base,iIdx,iFact, filterFlag)) + "\n")

    return pred_result

def calculate_pred_y(tb, x, iIdx, iFact, top_samples, left_samples, filter_table):

    fT = [0 for row in range(4)]
    for j in range(0, 4):
        fT[j] = gen.ft_coefficients[filter_table][str(iFact) + '[' + str(j) + ']']

    pred = 0
    for i in range(0, 4):
        ref = tb.ref[x + iIdx + i]
        index_x, index_y = re.findall(r'-?\d+', ref)  # get p[index] and ref from string containing and put it in two separately variables
        if int(index_x) == -1 and int(index_y) == -1:
            ref_value = top_samples[0]
        elif int(index_x) == -1:
            ref_value = left_samples[int(index_y)]
        elif int(index_y) == -1:
            ref_value = top_samples[int(index_x) + 1]
        else:
            throw_error("Undefined sample")

        pred += fT[i] * ref_value

    pred = clip1((pred + 32) >> 6)
    return pred


def clip1(value):
    if (value > 255):
        return 255

    if (value < 0):
        return 0

    return value


def automated_tests(seed, lower_limit, upper_limit):
    input = {}
    rm.seed(seed)  # set seed so that input has the same values for all tests
    fin = open(path_tests + "main_input.vhd", "w")
    # generate inputs
    for n in range(lower_limit, 0):
        input[(-1, -(n + 1))] = rm.randint(0, 255)
        fin.write("input(" + str(n) + ") <= \"" + str(np.binary_repr(input[(-1, -(n + 1))], width=8)) + "\"; --" + str(
            input[(-1, -(n + 1))]) + "\n")

    input[(-1, -1)] = rm.randint(0, 255)
    fin.write("input(" + str(0) + ") <= \"" + str(np.binary_repr(input[(-1, -1)], width=8)) + "\"; --" + str(
        input[(-1, -1)]) + "\n")

    for n in range(0, upper_limit):
        input[(n, -1)] = rm.randint(0, 255)
        fin.write("input(" + str(n + 1) + ") <= \"" + str(np.binary_repr(input[(n, -1)], width=8)) + "\"; --" + str(
            input[(n, -1)]) + "\n")

    fin.close()

    '''for key, value in zip(input.keys(), input.values()):
        print(key, value)'''

    return input


