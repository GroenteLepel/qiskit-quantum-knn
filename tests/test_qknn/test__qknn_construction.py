import os, logging

from qiskit import QuantumRegister

from qiskit_quantum_knn.config import Config
from qiskit_quantum_knn.qknn._qknn_construction import create_oracle

logging.basicConfig(level=logging.DEBUG)


def test_create_oracle(caplog):
    logger = logging.getLogger('__name__')
    print(os.getcwd())
    logger.error("test")
    create_oracle(
        r_train=QuantumRegister(2),
        r_comp_basis=QuantumRegister(2),
        train_data=
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]
    )
    assert "sopmething" in caplog.text