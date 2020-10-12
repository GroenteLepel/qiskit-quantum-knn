import os, logging

from qiskit import QuantumRegister

from qiskit_quantum_knn.qknn._qknn_construction import create_oracle


def test_create_oracle(caplog):
    caplog.set_level(logging.DEBUG)
    oracle = create_oracle(
        r_train=QuantumRegister(2),
        r_comp_basis=QuantumRegister(2),
        train_data=
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]
    )

    assert len(oracle.definition.qubits) == 4, \
        "train and computational basis registers of len 2 have not made an " \
        "oracle of len 4. "
