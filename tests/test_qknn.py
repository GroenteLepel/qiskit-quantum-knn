import unittest

import qiskit as qk
import numpy as np

from .context import qiskit_quantum_knn
import qiskit_quantum_knn.qknn._qknn_construction as constr

# TODO: find a proper place for these vectors
# quickly accessible train vector to goof around with
orthogonal_vect_1 = \
    [
        complex(0, 0),
        complex(0, 0),
        complex(1, 0),
        complex(0, 0),
        complex(0, 0),
        complex(0, 0),
        complex(0, 0),
        complex(0, 0)
    ]

orthogonal_vect_2 = \
    [
        complex(0, 0),
        complex(0, 0),
        complex(0, 0),
        complex(0, 0),
        complex(1, 0),
        complex(0, 0),
        complex(0, 0),
        complex(0, 0),
    ]

# quickly accessible "set" of data points to goof around with
orthogonal_data = \
    [
        orthogonal_vect_1, orthogonal_vect_2
    ]


def get_counts(circ: qk.QuantumCircuit):
    """
    Simulates the provided circuit using BasicAer's qasm_simulator
    and returns the counts.
    :param circ: qk.QuantumCircuit to draw the statevector from.
    :return: dict.
    """
    backend = qk.BasicAer.get_backend('qasm_simulator')
    job = qk.execute(circ, backend)
    results = job.result()

    return results.get_counts()


class TestQknn(unittest.TestCase):
    def test_create_oracle(self):
        """
        Test if two orthogonal vectors are generated with equal probability
        using the create_oracle method.
        :return:
        """
        # get dimensionality and length of the data
        log2_dimension = np.log2(len(orthogonal_data[0]))  # 4
        n_samples = len(orthogonal_data)  # 2
        log2_n_samples = np.log2(n_samples)  # 1
        # build the registers from these values
        # quantum
        state_reg = qk.QuantumRegister(log2_dimension)
        control_reg = qk.QuantumRegister(log2_n_samples)
        # classical
        state_meas = qk.ClassicalRegister(len(state_reg))
        control_meas = qk.ClassicalRegister(len(control_reg))

        # create the circuit
        test_circ = qk.QuantumCircuit(
            state_reg,
            control_reg,
            state_meas,
            control_meas
        )

        # bring control in superposition
        test_circ.h(control_reg)

        # create the oracle and append to circuit
        oracle = constr.create_oracle(state_reg, control_reg, orthogonal_data)
        test_circ.append(oracle, state_reg[:] + control_reg[:])
        # add measurements
        test_circ.measure(state_reg, state_meas)
        test_circ.measure(control_reg, control_meas)

        # get counts
        counts = get_counts(test_circ)
        counts_arr = np.array(list(counts.values()))
        total = counts_arr.sum()

        # analyse counts
        # check if probabilities are evenly divided
        for counts in counts_arr:
            self.assertAlmostEqual(1 / n_samples, counts / total, places=1)
