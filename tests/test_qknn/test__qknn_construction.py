import logging

from qiskit_quantum_knn.qknn._qknn_construction import *


def test_create_qknn(caplog):
    caplog.set_level(logging.DEBUG)
    create_qknn(
        state_to_classify=[1, 0, 0, 0],
        classified_states=[
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ],
        add_measurement=True
    )


def test_create_oracle(caplog):
    caplog.set_level(logging.DEBUG)
    oracle = create_oracle(
        train_data=
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]
    )

    assert len(oracle.definition.qubits) == 3, \
        "train and computational basis registers of size 1 and 2 have not made " \
        "an oracle of len 3."
