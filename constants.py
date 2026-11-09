
# Physical constants

# Boltzmann constant (J/K)
K_B = 1.380649e-23

# Atomic mass unit (kg)
ATOMIC_MASS_UNIT = 1.66053906660e-27


# Molecular masses (kg per particle)
GAS_MASSES = {
    "N2": 28.0134 * ATOMIC_MASS_UNIT,
    "O2": 31.9988 * ATOMIC_MASS_UNIT,
    "He": 4.002602 * ATOMIC_MASS_UNIT,
    "Ar": 39.948 * ATOMIC_MASS_UNIT
}


# Available gases
AVAILABLE_GASES = list(GAS_MASSES.keys())
