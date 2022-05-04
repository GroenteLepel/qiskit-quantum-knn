import numpy as np
import qiskit as qk
from qiskit.utils import QuantumInstance
import pytest

from qiskit_quantum_knn.qknn import QKNeighborsClassifier


def test_get_all_contrasts():
    assert True


def test_calculate_fidelities():
    ex_counts = {
        '00 0': 10,
        '00 1': 1,
        '01 0': 20,
        '01 1': 1,
        '10 0': 15,
        '10 1': 3,
        '11 0': 2,
        '11 1': 4,
    }
    fidelities = QKNeighborsClassifier.calculate_fidelities(ex_counts)
    print(fidelities)

    # no physically correct fidelities but
    np.testing.assert_allclose(
        fidelities,
        [0.78826531, 1.01785714, 0.66326531, 0.24489796]
    )


def test_calculate_contrasts():
    assert True


def test_setup_control_counts():
    assert True


def test_majority_vote_voting():
    qknn = QKNeighborsClassifier(n_neighbors=3)
    example_fidelities = np.array([
        [0.1, 0.9, 0.1, 0.9],
        [0.9, 0.1, 0.9, 0.1],
    ])
    example_labels = np.array([-1, 1, -1, 1])
    votes = qknn.majority_vote(
        labels=example_labels,
        fidelities=example_fidelities
    )
    np.testing.assert_array_equal(votes, [1, -1])


@pytest.mark.parametrize(
    "n_train_states,n_test_states,n_qubit_states",
    [(8, 1, 8), (8, 8, 8), (5, 8, 8)]
)
def test_majority_vote_result_shape(n_train_states: int,
                                    n_test_states: int,
                                    n_qubit_states: int):
    qknn = QKNeighborsClassifier(n_neighbors=3)
    example_fidelities = np.random.rand(n_test_states, n_qubit_states)

    example_labels = np.random.choice(
        [-1, 1],
        n_train_states
    )

    votes = qknn.majority_vote(
        labels=example_labels,
        fidelities=example_fidelities
    )
    assert votes.shape == (n_test_states,)


def test_qknn():
    backend = qk.BasicAer.get_backend('qasm_simulator')
    instance = QuantumInstance(backend, shots=10000)

    # initialising the qknn model
    qknn = QKNeighborsClassifier(
        n_neighbors=3,
        quantum_instance=instance
    )

    train_data = [
        [1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
        [1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
        [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
        [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]
    ]
    train_labels = [
        1,
        1,
        -1,
        -1
    ]
    test_data = [
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
        [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
    ]

    qknn.fit(train_data, train_labels)
    qknn_prediction = qknn.predict(test_data)
    np.testing.assert_array_equal(qknn_prediction, [1, -1])
