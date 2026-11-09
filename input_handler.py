
from constants import AVAILABLE_GASES


def get_user_input():

    # Select gas
    while True:
        gas = input("Select gas (N2, O2, He, Ar): ").strip()

        if gas in AVAILABLE_GASES:
            break
        else:
            print("Invalid gas. Please choose N2, O2, He, or Ar.")

    # Enter temperature
    while True:
        try:
            T = float(input("Enter temperature (K): "))

            if T > 0:
                break
            else:
                print("Temperature must be greater than 0 K.")

        except ValueError:
            print("Please enter a valid number.")

    # Enter lower velocity
    while True:
        try:
            v1 = float(input("Enter lower velocity v1 (m/s): "))

            if v1 >= 0:
                break
            else:
                print("Velocity must be non-negative.")

        except ValueError:
            print("Please enter a valid number.")

    # Enter upper velocity
    while True:
        try:
            v2 = float(input("Enter upper velocity v2 (m/s): "))

            if v2 > v1:
                break
            else:
                print("v2 must be greater than v1.")

        except ValueError:
            print("Please enter a valid number.")

    return gas, T, v1, v2
