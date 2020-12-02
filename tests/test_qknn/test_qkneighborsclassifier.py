import numpy as np
import pytest

from qiskit_quantum_knn.qknn import QKNeighborsClassifier


def test_get_all_contrasts():
    assert True


def test_calculate_contrasts():
    assert True


def test_setup_control_counts():
    assert True


@pytest.mark.parametrize(
    "n_train_states,n_test_states,n_qubit_states",
    [(8, 1, 8), (8, 8, 8), (5, 8, 8)]
)
def test_majority_vote(n_train_states: int,
                       n_test_states: int,
                       n_qubit_states: int):
    if not n_qubit_states:
        n_qubit_states = n_train_states

    qknn = QKNeighborsClassifier(n_neighbors=3)
    example_contrasts = np.random.normal(
        0,
        size=(n_test_states, n_qubit_states)
    )
    example_labels = np.random.choice(
        [-1, 1],
        n_train_states
    )

    votes = qknn.majority_vote(
        labels=example_labels,
        contrasts=example_contrasts
    )
    assert votes.shape == (n_test_states,)