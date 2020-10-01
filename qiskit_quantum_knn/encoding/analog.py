import numpy as np

"""Script encoding classical data to quantum state via amplitude encoding."""


def encode(classical_data):
    """Encodes the given classical state to a quantum state.

    Args:
        classical_data (vector_like): state(s) to encode.

    Returns:
        np.ndarray: the encoded quantum state.
    """
    # sum up every row of the matrix to get the lengths for each row via
    #  a_ij * a_ij = A_i
    amplitudes = np.sqrt(np.einsum('ij,ij->i', classical_data, classical_data))

    # set zero amplitudes to 1 to prevent division through zero
    amplitudes[amplitudes == 0] = 1

    # normalise the data by dividing the original through the amplitude
    normalised_data = classical_data / amplitudes[:, np.newaxis]

    return normalised_data

