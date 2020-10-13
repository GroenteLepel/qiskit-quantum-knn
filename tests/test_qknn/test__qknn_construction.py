import logging

import pytest

from qiskit_quantum_knn.qknn._qknn_construction import *


def test_create_qknn_2_training_states(caplog):
    caplog.set_level(logging.DEBUG)
    qknn = create_qknn(
        state_to_classify=[1, 0, 0, 0],
        classified_states=
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ],
        add_measurement=False
    )
    assert qknn.num_qubits == 6, \
        "qknn with two training states of dim 4 did not create the expected 6" \
        " qubits qknn. (1 for control, 2 x 2 for states, and 1 for " \
        "computational basis) "


def test_create_oracle_2_training_states(caplog):
    caplog.set_level(logging.DEBUG)
    oracle = create_oracle(
        train_data=
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]
    )

    assert oracle.num_qubits == 3, \
        "Train and computational basis registers of size 1 and 2 have not " \
        "made an oracle with 3 qubits."


@pytest.mark.filterwarnings("ignore:Number of training states")
def test_create_oracle_uneven_train_states(caplog):
    caplog.set_level(logging.DEBUG)
    oracle = create_oracle(
        train_data=
        [
            [1, 0],
            [1, 0],
            [1, 0]
        ]
    )

    assert oracle.num_qubits == 3, \
        "Train and computational basis registers of size 1 and 3 have not " \
        "made an oracle with 4 qubits."


def test_create_oracle_uneven_dim(caplog):
    with pytest.raises(Exception):
        caplog.set_level(logging.DEBUG)
        create_oracle(
            train_data=
            [
                [1, 0, 0]
            ]
        )


def test_where_to_apply_x():
    assert where_to_apply_x(2) == [[0, 1], [0], [0, 1], [0]]
