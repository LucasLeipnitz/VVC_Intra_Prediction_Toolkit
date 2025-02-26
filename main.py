import generator as gen
import verifier as ver

path_input_modes = "./input/modes/"

option = 0
modes = gen.modes4
block_size = 8
assert_equals = 1
normalize = 1
heuristic_on = True

def main(modes, control = -1):
    
    angles = gen.map_modes_to_angles(modes)

    gen.transform_coefficients(8, 16,False, False)

    if control == 0:
        gen.calculate_equations(modes, angles, block_size, "fc_heuristic")
    elif control == 1:
        gen.calculate_samples(modes, angles, block_size, normalize=normalize)
    elif control == 2:
        gen.calculate_iidx_ifact(modes, angles, block_size)
    elif(control == 3):
        if(assert_equals):
            input = ver.automated_tests(196, -66, 38)
            ver.assert_equals_planar(input, 32)
            ver.assert_equals_dc(input, 32, 135)
            for i in range (0,16):
                ver.assert_equals_angular(input, modes[i], angles[i], block_size, -66, 38)
        else:
            ver.automated_tests(196, -66, 38)

    else:
        print("Select a value for control between 0 and 3")
    

if __name__ == "__main__":
    main(modes, control = option)


