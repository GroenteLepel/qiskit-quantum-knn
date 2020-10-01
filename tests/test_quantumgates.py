import unittest

import qiskit.quantum_info.states.statevector as sv
import numpy as np
import qiskit as qk

from .context import qiskit_quantum_knn
import qiskit_quantum_knn.qknn.quantumgates as gates

# TODO: find a proper place for this vector and method
superposition_state = \
    [
        complex(1/np.sqrt(3), 0),
        complex(0, 0),
        complex(1/np.sqrt(3), 0),
        complex(0, 0),
        complex(1/np.sqrt(3), 0),
        complex(0, 0),
        complex(0, 0),
        complex(0, 0)
    ]


def get_statevector(circ: qk.QuantumCircuit):
    """
    Simulates the provided circuit using BasicAer's statevector_simulator
    and returns the statevector.
    :param circ: qk.QuantumCircuit to draw the statevector from.
    :return: ndarray (complex).
    """
    backend = qk.BasicAer.get_backend('statevector_simulator')
    job = qk.execute(circ, backend)
    results = job.result()

    return results.get_statevector()


class Test(unittest.TestCase):
    def test_init_to_state(self):
        """
        Tests if the method init_to_state() properly initialises a
        QuantumCircuit to the provided statevector.
        :return:
        """
        state_to_init = sv.Statevector(superposition_state)
        n_qubits = np.log2(len(superposition_state))

        state_register = qk.QuantumRegister(n_qubits)
        circ = qk.QuantumCircuit(state_register)

        init_gate = gates.init_to_state(state_register, superposition_state)
        circ.append(init_gate, state_register)
        measured_state_vector = get_statevector(circ)

        self.assertAlmostEqual(
            (state_to_init - measured_state_vector).data.sum(),
            0
        )
