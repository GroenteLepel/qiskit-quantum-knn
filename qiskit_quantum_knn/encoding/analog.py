import numpy as np

"""Script encoding classical data to quantum state via amplitude encoding."""


def encode(classical_data):
    r"""Encodes the given classical state to a quantum state.

    The encoding of a state makes sure that the following rule within quantum
    mechanics is met:

    .. math:: \langle \psi | \psi \rangle = 1,

    for a quantum state :math:`\psi`. To put it in other words, suppose that
    :math:`\psi = \sum_i c_i x_i` with :math:`c_i \in \mathbb{C}`, the encoding
    will normalise the values :math:`c_i` such that :math:`\sum_i |c_i|^2 = 1`.

    Example:

        Simple encoding using real values.

        .. jupyter-execute::

            from qiskit_quantum_knn.encoding.analog import encode

            classical_state = [
                [1, 1, 1, 1]
            ]
            normalised_state = encode(classical_state)

            print(normalised_state)
            print((normalised_state ** 2).sum())

        Using complex values.

        .. jupyter-execute::

            classical_state = [
                [1+2j, 1-3j, 1+2j, 1+3j]
            ]
            normalised_state = encode(classical_state)

            print(normalised_state)
            print((normalised_state ** 2).sum())

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

