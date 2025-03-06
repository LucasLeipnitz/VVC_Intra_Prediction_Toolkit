import generator as gen
import verifier as ver

path_input_modes = "./input/modes/"

option = 3
##modes = gen.all_modes
input_modes = [27]
block_size = 4
assert_equals = 1
normalize = 1
heuristic_on = True

def main(modes, control = -1):
    
    angles = gen.map_modes_to_angles(modes)

    gen.transform_coefficients(8, 8,False, False)

    match control:
        case 0:
            for mode, angle in zip(modes, angles):
                equations, equations_set = gen.calculate_equations(mode, angle, block_size, "fc_heuristic", 1)
                gen.generate_sorted_equations_set(equations_set, True)
        case 1:
            gen.calculate_samples(modes, angles, block_size, normalize=normalize)
        case 2:
            gen.calculate_iidx_ifact(modes, angles, block_size)
        case 3:
            for mode, angle in zip(modes, angles):
                equations, equations_set = gen.calculate_equations(mode, angle, block_size, "fc_heuristic", 1)
                sorted_equations_set = gen.generate_sorted_equations_set(equations_set, True)
                gen.generate_control_sequence(sorted_equations_set,True)
        case 4:
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


