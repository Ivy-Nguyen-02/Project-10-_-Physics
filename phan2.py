import numpy as np

# Boltzmann constant (J/K)
k_B = 1.380649e-23


def maxwell_boltzmann(v, T, m):
    """
    Calculate the Maxwell-Boltzmann speed distribution.

    Parameters
    ----------
    v : float or numpy array
        Particle speed (m/s).
    T : float
        Absolute temperature (K).
    m : float
        Mass of one particle (kg).

    Returns
    -------
    f : float or numpy array
        Probability density (s/m).
    """
    coefficient = 4 * np.pi * (m / (2 * np.pi * k_B * T))**1.5
    f = coefficient * v**2 * np.exp(
        -m * v**2 / (2 * k_B * T)
    )
    return f



