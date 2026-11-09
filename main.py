
from constants import K_B, GAS_MASSES
from input_handler import get_user_input


def main():

    # Get input from user
    gas, T, v1, v2 = get_user_input()

    # Get molecular mass of selected gas
    mass = GAS_MASSES[gas]

    # Display input information
    print("\n========== PROJECT INFORMATION ==========")

    print(f"Selected gas: {gas}")
    print(f"Temperature: {T} K")
    print(f"Lower velocity: {v1} m/s")
    print(f"Upper velocity: {v2} m/s")

    print(f"Boltzmann constant: {K_B} J/K")
    print(f"Molecular mass: {mass:.6e} kg")

    print("==========================================")


if __name__ == "__main__":
    main()

