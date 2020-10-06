from qiskit_quantum_knn.encoding.analog import encode


def test_encode_with_four_ones():
    assert (encode([[1, 1, 1, 1]]) == [1/2, 1/2, 1/2, 1/2]).all(), "not normalising properly with vector of ones."