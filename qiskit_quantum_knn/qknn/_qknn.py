import logging

from qiskit.aqua.algorithms.classifiers.qsvm._qsvm_abc import _QSVM_ABC

logger = logging.getLogger(__name__)


class _QKNN(_QSVM_ABC):
    """The qknn classifier.

    A class maintaining:
        - a QKNeighborsClassifier quantum algorithm;
        - manages the running, testing and predicting using all available data
        in said quantum algorithm.
    """

    def __init__(self, qalgo):
        super().__init__(qalgo)

    def predict(self, data):
        circuits = self._qalgo.construct_circuits(
            data,
            self._qalgo.training_dataset,
        )
        circuit_results = self._qalgo.get_circuit_results(
            circuits
        )
        fidelities = self._qalgo.get_all_fidelities(
            circuit_results
        )

        predicted_labels = self._qalgo.majority_vote(
            self._qalgo.training_labels,
            fidelities
        )

        return predicted_labels

    def run(self):
        circuits = self._qalgo.construct_circuits(
            self._qalgo.data_points,
            self._qalgo.training_dataset,
        )
        circuit_results = self._qalgo.get_circuit_results(
            circuits
        )
        fidelities = self._qalgo.get_all_fidelities(
            circuit_results
        )

        predicted_labels = self._qalgo.majority_vote(
            self._qalgo.training_labels,
            fidelities
        )

        self._ret['counts'] = circuit_results.get_counts()
        self._ret['fidelities'] = fidelities
        self._ret['predicted_labels'] = predicted_labels

    @staticmethod
    def execute_all(qalgo, data, training_data):
        pass