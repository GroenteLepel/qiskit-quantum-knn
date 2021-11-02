"""The quantum KNN algorithm."""

from typing import Dict, Optional, Union
import logging
import itertools
import numbers

import numpy as np
import scipy.stats as stats
import qiskit as qk
import qiskit.result as qres
import qiskit.tools as qktools
import qiskit.aqua as aqua
import qiskit.providers as qkp
import qiskit.aqua.utils.subsystem as ss
from qiskit.aqua.algorithms import QuantumAlgorithm
import qiskit.circuit.instruction as qinst

from qiskit_quantum_knn.qknn._qknn import _QKNN
from qiskit_quantum_knn.qknn import qknn_construction as qc

logger = logging.getLogger(__name__)

UnionQInstBaseB = Union[aqua.QuantumInstance, qkp.BaseBackend]
OptionalQInstance = Optional[UnionQInstBaseB]


class QKNeighborsClassifier(QuantumAlgorithm):
    """ Quantum KNN algorithm.

    Maintains the construction of a QkNN Quantumcircuit, and manages the data
    corresponding with this circuit by setting up training and test data and
    maintaining the classes and labels to the data.

    Args:
        n_neighbors (int): number of neighbors to perform the voting.
            training_dataset (array-like): data shaped ``(n, d)``, with ``n``
            the number of data points, and ``d`` the dimensionality.
            Corresponds to the training data, which is classified and will be
            used to classify new data. ``d`` must be a positive power of two,
            ``n`` not per se, because it can be zero-padded to fit on a quantum
            register.
        training_labels (array): the labels corresponding to the training data,
            must be ``len(n)``.
        test_dataset (array-like): data shaped ``(m, d)``, with ``m`` the
            the number of data points, and ``d`` the dimensionality. Describes
            test data which is used to test the algorithm and give an
            accuracy score.

            TODO: this is not implemented yet, for now a test is performed manually.

        data_points (array-like): data shaped ``(k, d)``, with ``k`` the number
            of data points, and ``d`` the dimensionality of the data. This is
            the unlabelled data which  must be classified by the algorithm.
        quantum_instance (:class: `QuantumInstance` or :class: BaseBackend):
            the instance which ``qiskit`` will use to run the quantum algorithm.

    Example:

        Classify data using the Iris dataset.

        .. jupyter-execute::

            from qiskit_quantum_knn.qknn import QKNeighborsClassifier
            from qiskit_quantum_knn.encoding import analog
            from qiskit import aqua
            from sklearn import datasets
            import qiskit as qk

            # initialising the quantum instance
            backend = qk.BasicAer.get_backend('qasm_simulator')
            instance = aqua.QuantumInstance(backend, shots=10000)

            # initialising the qknn model
            qknn = QKNeighborsClassifier(
               n_neighbors=3,
               quantum_instance=instance
            )

            n_variables = 2        # should be positive power of 2
            n_train_points = 4     # can be any positive integer
            n_test_points = 2      # can be any positive integer

            # use iris dataset
            iris = datasets.load_iris()
            labels = iris.target
            data_raw = iris.data

            # encode data
            encoded_data = analog.encode(data_raw[:, :n_variables])

            # now pick these indices from the data
            train_data = encoded_data[:n_train_points]
            train_labels = labels[:n_train_points]

            test_data = encoded_data[n_train_points:(n_train_points+n_test_points), :n_variables]
            test_labels = labels[n_train_points:(n_train_points+n_test_points)]

            qknn.fit(train_data, train_labels)
            qknn_prediction = qknn.predict(test_data)

            print(qknn_prediction)
            print(test_labels)

    """

    def __init__(self, n_neighbors: int = 3,
                 training_dataset: Optional[np.ndarray] = None,
                 training_labels: Optional[np.ndarray] = None,
                 test_dataset: Optional[np.ndarray] = None,
                 data_points: Optional[np.ndarray] = None,
                 quantum_instance: OptionalQInstance = None) -> None:
        super().__init__(quantum_instance)

        self.n_neighbors = n_neighbors

        # the datasets for training, testing and predicting
        self.training_dataset = None
        self.test_dataset = None
        self.data_points = None
        # dictionaries containing the class vs labels and reverse, respectively
        self.class_to_label = None
        self.label_to_class = None
        # the number of classes in the provided data sets
        self.num_classes = None

        # setup of the data
        self.training_dataset = training_dataset
        self.training_labels = training_labels
        self.test_dataset = test_dataset
        self.data_points = data_points

        # indicates the kind of instance this classifier is using, e.g.
        #  _QSVM_Binary, _QSVM_Multiclass, or, in this case,  _QKNN. This
        #  instance has all the methods for running, testing and predicting.
        self.instance = _QKNN(self)

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Notes:
            There is no real "fitting" done here, since the data cannot
            be stored somewhere. It only assigns the values so that these
            cane be accessed when running.
        Args:
            X (array-like): Training data of shape [n_samples, n_features].
            y (array-like): Target values of shape [n_samples].
        """
        # TODO: create some validation and checks for the provided data

        self.training_dataset = X
        self.training_labels = y

    @staticmethod
    def construct_circuit(state_to_classify: np.ndarray,
                          oracle: qinst.Instruction,
                          add_measurement: bool = False) -> qk.QuantumCircuit:
        r"""Construct one QkNN QuantumCircuit.

        The Oracle provided is mentioned in :afham2020:`Afham et al. (2020)`
        as the parameter :math:`\mathcal{W}`, and is created via the method
        :py:func:`~qiskit_quantum_knn.qknn.qknn_construction.create_oracle`.

        Args:
            state_to_classify (array-like): array of dimension ``N`` complex
                values describing the state to classify via kNN.
            oracle (qiskit Instruction): oracle :math:`\mathcal{W}` for applying
                training data.
            add_measurement (bool): controls if measurements must be added
                to the classical registers.

        Returns:
            QuantumCircuit: The constructed circuit.
        """
        return qc.construct_circuit(
            state_to_classify,
            oracle,
            add_measurement
        )

    @staticmethod
    def construct_circuits(data_to_predict,
                           training_data) -> qk.QuantumCircuit:
        """Constructs all quantum circuits for each datum to classify.

        Args:
            data_to_predict (array): data points, 2-D array, of shape
                ``(N, D)``, where ``N`` is the number of data points and ``D``
                is the dimensionality of the vector. ``D`` should coincide with
                the provided training data.
            training_data (array): data points which you want to know
                the distance of between :py:attr:`data_to_predict`.

        Returns:
            numpy.ndarray: The constructed circuits.

        Raises:
            AquaError: Quantum instance is not present.

        """
        measurement = True  # can be adjusted if statevector_sim
        oracle = qc.create_oracle(training_data)

        # parallel_map() creates QuantumCircuits in parallel to be executed by
        #  a QuantumInstance
        logger.info("Starting parallel map for constructing circuits.")
        circuits = qktools.parallel_map(
            QKNeighborsClassifier.construct_circuit,
            data_to_predict,
            task_args=[
                oracle,
                measurement
            ]
        )
        logger.info("Done.")

        return circuits

    @staticmethod
    def execute_circuits(quantum_instance: UnionQInstBaseB,
                         circuits) -> qres.Result:
        """Executes the provided circuits (type array-like)."""
        logger.info("Executing circuits")
        result = quantum_instance.execute(circuits)
        logger.info("Done.")
        return result

    def get_circuit_results(self,
                            circuits,
                            quantum_instance: OptionalQInstance = None) -> qres.Result:
        """Get the qiskit Results from the provided quantum circuits."""
        self._quantum_instance = self._quantum_instance \
            if quantum_instance is None else quantum_instance
        if self._quantum_instance is None:
            raise aqua.AquaError(
                "Either provide a quantum instance or set one up."
            )

        return QKNeighborsClassifier.execute_circuits(
            self.quantum_instance,
            circuits
        )

    @staticmethod
    def get_all_fidelities(circuit_results: qres.Result):
        r"""Get all contrasts.

        Gets the fidelity values which are calculated via
        :func:`calculate_fidelities` and saves these in an array. For more about
        fidelities, see :meth:`calculate_fidelities`.

        Args:
            circuit_results (qiskit.result.Result): the results from a QkNN
                circuit build using ``QKNeighborsClassifier``.

        Returns:
            array: all fidelities corresponding to the QkNN.
        """
        logger.info("Getting fidelity values.")
        # get all counts from the circuit results
        all_counts = circuit_results.get_counts()
        # determine the length of the computational basis register by checking
        #  the length of the count result
        # the -2 is there to compensate for the ' 0' or ' 1' at the end of
        #  the key.
        num_qubits = len(list(all_counts[0].keys())[0]) - 2

        # initialize the array which will hold the contrast values
        n_occurrences = len(all_counts)  # number of occurring states
        n_datapoints = 2 ** num_qubits  # number of data points

        all_fidelities = np.empty(
            shape=(n_occurrences, n_datapoints),
        )

        # loop over all counted states
        for i, counts in enumerate(all_counts):
            # calculate the contrast values q(i) for this set of counts
            all_fidelities[i] = \
                QKNeighborsClassifier.calculate_fidelities(counts)
        logger.info("Done.")

        return all_fidelities

    @staticmethod
    def calculate_fidelities(counts: Dict[str, int]) -> np.ndarray:
        r"""Calculate the fidelities :math:`F_i`.

        Calculates fidelities :math:`F_i` for each training state ``i`` in the
        computational basis of the kNN QuantumCircuit. The fidelity can be
        calculated via:

        .. math::

            F_i = \frac{M}{2} \left(p_0 (i) - p_1 (i)\right) \cdot \
                \left(1 - \left( p(0) - p(1) \right) ^2 \right) + \
                \left( p(0) - p(1) \right).

        The values :math:`p(n)` are the probabilities that the control qubit is
        in state :math:`n`, and the values :math:`p_n (i)` are the probabilities
        that the computational basis is in state :math:`i` given the control
        qubit is in state :math:`n`.

        These values can be approximated by running the circuit :math:`T`
        times using:

        .. math::
            p_n (i) \sim \bar{p}_n (i) = c_n(i) / T_n , \
            p (n) \sim \bar{p} (n) = T_n / T,

        where :math:`c_n(i), T_n` are the counts of the computational basis
        in state :math:`i` given the control qubit in state :math:`n` and the
        control qubit in state :math:`n`, respectively.

        Args:
            counts (dict): counts pulled from a qiskit Result from the QkNN.

        Returns:
            array: the fidelity values.

            ndarray of length ``n_samples`` with each index ``i`` (representing
            state :math:`|i\rangle` from the computational basis) the fidelity
            belonging to :math:`|i\rangle`.
        """
        # first get the total counts of 0 and 1 in the control qubit
        subsystem_counts = ss.get_subsystems_counts(counts)
        # the counts from the control qubit are in the second register
        #  by some magical qiskit reason
        control_counts = QKNeighborsClassifier.setup_control_counts(
            subsystem_counts[1]
        )
        total_counts = control_counts['0'] + control_counts['1']
        exp_fidelity = np.abs(control_counts['0'] - control_counts['1']) / \
            total_counts

        # now get the counts for the fidelities define possible states that
        #  the computational can be in.
        num_qubits = len(list(subsystem_counts[0].keys())[0])
        comp_basis_states = \
            list(itertools.product(['0', '1'], repeat=num_qubits))
        # initialise dict which is going to contain the fidelity values
        fidelities = np.zeros(2 ** num_qubits, dtype=float)
        for comp_state in comp_basis_states:
            # convert list of '0's and '1's to one string e.g.
            #  ('0', '1', '0') --> '010'
            comp_state = ''.join(comp_state)
            # init fidelity value for this state
            fidelity = 0.
            for control_state in control_counts.keys():
                state_str = comp_state + ' ' + control_state
                if state_str not in counts:
                    logger.debug(
                        "State {0:s} not found in counts {1}. Adding"
                        "naught to contrast value."
                        .format(
                            state_str,
                            counts
                        )
                    )
                    fidelity += 0  # added for readability
                else:
                    fidelity += \
                        (-1) ** int(control_state) * \
                        (counts[state_str]) / control_counts[control_state] * \
                        (1 - exp_fidelity ** 2)
            index_state = int(comp_state, 2)
            fidelity *= 2 ** num_qubits / 2
            fidelity += exp_fidelity
            fidelities[index_state] = fidelity
        return fidelities

    @staticmethod
    def calculate_contrasts(counts: Dict[str, int]) -> np.ndarray:
        r"""Calculate contrasts :math:`q(i)`.

        Calculates contrasts :math:`q(i)` for each training state ``i`` in the
        computational basis of the KNN QuantumCircuit. The contrasts
        are according to :afham2020:`Afham et al. (2020)`.

        .. math::

           q(i) &= p_0(i) - p_1(i) \\
                &= \frac{1 + F_i}
                    {M + \sum_{j=1}^M F_j} - \
                    \frac{1 - F_i}
                    {M - \sum_{j=1}^M F_j} \\
                &= \frac{2(F_i - \langle F \rangle)}
                    {M(1 - \langle F \rangle^2)},

        and correspond linearly to the fidelity :math:`F_i` between the
        unclassified datum :math:`\psi` and :math:`\phi_i`.

        Args:
            counts (dict): counts pulled from a qiskit Result from the QkNN.

        Returns:
            array: the contrasts values.

            ndarray of length ``n_samples`` with each index ``i`` (representing
            state :math:`|i\rangle` from the computational basis) the contrast
            belonging to :math:`|i\rangle`.

        """
        # first get the total counts of 0 and 1 in the control qubit
        subsystem_counts = ss.get_subsystems_counts(counts)
        # the counts from the control qubit are in the second register
        #  by some magical qiskit reason
        control_counts = QKNeighborsClassifier.setup_control_counts(
            subsystem_counts[1]
        )

        # now get the counts for the contrasts define possible states that
        #  the computational can be in.
        num_qubits = len(list(subsystem_counts[0].keys())[0])
        comp_basis_states = \
            list(itertools.product(['0', '1'], repeat=num_qubits))
        # initialise dict which is going to  contain the contrast values
        contrasts = np.zeros(2 ** num_qubits, dtype=float)
        for comp_state in comp_basis_states:
            # convert list of '0's and '1's to one string e.g.
            #  ('0', '1', '0') --> '010'
            comp_state = ''.join(comp_state)
            # init contrast value for this state
            contrast = 0.
            for control_state in control_counts.keys():
                state_str = comp_state + ' ' + control_state
                if state_str not in counts:
                    logger.debug(
                        "State {0:s} not found in counts {1}. Adding"
                        "naught to contrast value."
                        .format(
                            state_str,
                            counts
                        )
                    )
                    contrast += 0  # added for readability
                else:
                    contrast += \
                        (-1) ** int(control_state) * \
                        (counts[state_str]) / control_counts[control_state]
            index_state = int(comp_state, 2)
            contrasts[index_state] = contrast

        return contrasts

    @staticmethod
    def setup_control_counts(control_counts: Dict[str, int]) -> Dict[str, int]:
        """Sets up control counts dict.

        In Qiskit, if a certain value is not measured (or counted), it has
        no appearance in the ``counts`` dictionary from the :py:class:`Result`.
        Thus, this method checks if this has happened and adds a value with
        counts set to 0.

        Notes:
            This means that if the :py:attr:`control_counts` has both
            occurrences of ``0`` and ``1``, this method just returns that exact
            same dictionary, unmodified.

        Args:
            control_counts (dict): dictionary from a :py:class:`Result`
            representing the control qubit in the QkNN circuit.

        Returns:
            dict: The modified control counts.

                The same control_counts dict as provided but with non-counted
                occurrence added as well if needed.

        Raises:
            ValueError: if the provided dictionary does not coincide with the
                :py:class:`Result` from the QkNN.
        """
        # constant describing the states possible in control_counts
        control_states = np.array(['0', '1'])
        # check if substition of 0 count value must be done
        if control_states[0] not in control_counts:
            to_substitute = int(control_states[0])
        elif control_states[1] not in control_counts:
            to_substitute = int(control_states[1])
        else:
            to_substitute = None

        if to_substitute is not None:
            # if to_substitute = 1, make it -1 * (1 - 1) = 0, else, make it
            #  -1 * (0 - 1) = 1
            sole_occurrence = -1 * (to_substitute - 1)
            logger.debug(
                "Only one value is counted in the control qubit: {0:d},"
                "setting the counts of state {1:d} to 0."
                    .format(
                        sole_occurrence,
                        to_substitute
                    )
            )
            control_counts = {
                str(to_substitute): 0,
                str(sole_occurrence): control_counts[str(sole_occurrence)]
            }

        return control_counts

    def majority_vote(self,
                      labels: np.ndarray,
                      fidelities: np.ndarray) -> np.ndarray:
        """Performs majority vote with the :math:`k` nearest to determine class.

        Args:
            labels (array-like): The labels of the training data provided to
                the :class:`QKNeighborsClassifier`.
            fidelities (array-like): The fidelities calculated using
                :meth:`get_all_fidelities'.

        Returns:
            ndarray: The labels resulted from the majority vote.
        """
        logger.info("Performing majority vote.")
        # get the neighbors sorted on their distance (lowest first) per data
        #  point.
        if np.any(fidelities < -0.2) or np.any(fidelities > 1.2):
            raise ValueError("Fidelities contain values outside range 0<=F<=1:"
                             f"{fidelities[fidelities < -0.2]}"
                             f"{fidelities[fidelities > 1.2]}")

        sorted_neighbors = np.argpartition(
            1 - fidelities,
            -self.n_neighbors
        )
        # get the number of participating values
        n_queries = len(labels)

        # modify the argpartition to remove any "filler" qubits, e.g. if 5
        #  train data are given, n_queries=5 but number of qubits states must
        #  always be a positive number of 2 (will be 8 in example case)
        # these values can accidentally participate in the voting, hence these
        #  must be removed
        sorted_neighbors = sorted_neighbors[sorted_neighbors < n_queries]\
            .reshape(sorted_neighbors.shape[0], n_queries)

        if n_queries == 1:
            n_closest_neighbors = sorted_neighbors[:self.n_neighbors]
        else:
            # this is the case when more than one data point is given to this
            #  majority vote, so the shape will be of (n_points, m)
            n_closest_neighbors = sorted_neighbors[:, :self.n_neighbors]

        # voters = np.take(data, indices_of_neighbors, axis=0)
        voter_labels = np.take(labels, n_closest_neighbors)
        if n_queries == 1:
            votes, counts = stats.mode(voter_labels)
        else:
            votes, counts = stats.mode(voter_labels, axis=1)

        logger.info("Done.")
        return votes.real.flatten()

    @property
    def ret(self) -> Dict:
        """ Returns result.

        Returns:
            Dict: return value(s).
        """
        return self.instance.ret

    @ret.setter
    def ret(self, new_value):
        """ Sets result.

        Args:
            new_value: new value to set.
        """
        self.instance.ret = new_value

    def predict(self, data) -> np.ndarray:
        """Predict the labels of the provided data."""
        return self.instance.predict(data)

    def _run(self) -> Dict:
        return self.instance.run()
