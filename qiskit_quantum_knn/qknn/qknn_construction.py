"""Construction of a QkNN QuantumCircuit."""

import logging, warnings
from typing import List, Union

import numpy as np
import qiskit as qk
import qiskit.extensions.quantum_initializer as qi
import qiskit.circuit as qcirc
import qiskit.circuit.instruction as qinst

import qiskit_quantum_knn.qknn.quantumgates as gates

logger = logging.getLogger(__name__)


def create_qknn(state_to_classify: Union[List, np.ndarray],
                classified_states: Union[List, np.ndarray],
                add_measurement: bool = False) -> qk.QuantumCircuit:
    """ Construct one QkNN QuantumCircuit.

    This method creates a circuit to perform distance measurements
    using quantum fidelity as distance metric :afham2020:`Afham et al.
    (2020)`. It initialises one register
    with a state to classify, and uses an Oracle to act as QRAM to
    hold the training data. This Oracle writes all training data in
    superposition to a register. After that, a swap-test circuit
    :buhrman2020:`Buhrman et al. (2020)` is created to perform the fidelity
    measurement.

    Example:
        Creating a circuit with simple data.

        .. jupyter-execute::

            from qiskit_quantum_knn.qknn.qknn_construction import create_qknn

            test_data = [1, 0]

            train_data = [
                [1, 0],
                [0, 1]
            ]

            circuit = create_qknn(test_data, train_data, add_measurement=True)
            print(circuit.draw())

    Args:
        state_to_classify (numpy.ndarray): array of dimension N complex
            values describing the state to classify via KNN.
        classified_states (numpy.ndarray): array containing M training
            samples of dimension N.
        add_measurement (bool): controls if measurements must be added
            to the classical registers.

    Returns:
        QuantumCircuit: the constructed circuit.

    """
    oracle = create_oracle(classified_states)
    return construct_circuit(
        state_to_classify,
        oracle,
        add_measurement
    )


def construct_circuit(state_to_classify: np.ndarray,
                      oracle: qinst.Instruction,
                      add_measurement: bool) -> qk.QuantumCircuit:
    r"""Setup for a QkNN QuantumCircuit.

    Constructs the QkNN QuantumCircuit according to the stepwise "instructions"
    in :afham2020:`Afham et al. (2020)`. These instructions are\:

    1. Initialisation:
        creates the registers and applies the unclassified
        datum :math:`\psi` (see :py:func:`initialise_qknn`);
    2. State transformation:
        applies :math:`H`-gates and the Oracle
        :math:`\mathcal{W}` to the circuit, and applies the
        :math:`SWAP`-test (see :py:func:`state_transformation`);
    3. Adding measurments:
        add the measurement gates to the control and
        computational basis (see :py:func:`add_measurements`).

    Args:
        state_to_classify (numpy.ndarray): array of dimension N complex
            values describing the state to classify via KNN.
        oracle (qiskit Instruction): oracle :math:`\mathcal{W}` for applying
            training data.
        add_measurement (bool): controls if measurements must be added
            to the classical registers.

    Raises:
        ValueError: If the number of data points in :attr:`state_to_classify`
            is more than 2.
        ValueError: If the length of the vectors in the
            :attr:`classified_states` and/or test data are not a positive
            power of 2.

    Returns:
        QuantumCircuit: constructed circuit.
    """

    if len(np.array(state_to_classify).shape) != 1:
        raise ValueError(
            f"Please only one data point to classify. Number of data points "
            f"provided is: {np.array(state_to_classify).shape[0]}. "
        )

    # get the dimensions of the state to classify
    state_dimension = len(state_to_classify)

    # get the number of qubits for the registers containing the state to
    #  classify and the number of train samples
    n = np.log2(state_dimension)  # n qubits for describing states
    m = oracle.num_qubits - n  # n qubits for computational basis

    # Check if param is a power of 2
    if (n == 0 or not n.is_integer()) and (m == 0 or not m.is_integer()):
        raise ValueError(
            "Desired statevector length not a positive power of 2."
        )

    # step 1: initialise (creates registers, sets qubits to |0> or the
    #  state to classify
    qknn_circ = initialise_qknn(n, m, state_to_classify)
    # step 2: state trans. (applies oracle)
    qknn_circ = state_transformation(qknn_circ, oracle)
    # step 3: adds the measurement gates
    if add_measurement:
        qknn_circ = add_measurements(qknn_circ)

    logger.debug(f"Final circuit:\n{qknn_circ.draw(fold=90)}")

    return qknn_circ


# noinspection PyTypeChecker
def create_oracle(train_data: Union[List, np.ndarray]) -> qinst.Instruction:
    r"""Create an Oracle to perform as QRAM.


    The oracle works as follows\:

    .. math:: \mathcal{W}|i\rangle |0\rangle = |i\rangle |\phi_i\rangle

    where the equation is from :afham2020:`Afham et al. (2020)`. This oracle
    acts as QRAM, which holds the training dataset :math:`\Phi` to assign to
    the register for performing a swap test. It is located in the center of
    the quantum circuit (see :py:func:`create_qknn`).

    Notes:
        The Oracle works with controlled initializers which check the
        state of the computational basis. The computational basis is described
        by :math:`|i\rangle`, where :math:`i` is any real number, which is then
        described by qubits in binary.

        To check the the state of the computational basis, a network of
        :math:`X`-gates is created to bring the computational basis
        systematically into all possible states. If all qubits in the register
        are :math:`|1\rangle`, the datum is assigned via the initialize. Where
        to apply the :math:`X`-gates is determined by
        :py:func:`where_to_apply_x`.

    Example:
        Creating a simple oracle for dataset with 4 points.

        .. jupyter-execute::

               from qiskit_quantum_knn.qknn.qknn_construction import create_oracle

               train_data = [
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1]
               ]

               oracle = create_oracle(train_data)

               print(oracle.definition.draw())

    Args:
        train_data (array-like): List of vectors with dimension
            ``len(r_train)`` to initialize ``r_train`` to.

    Returns:
        circuit.instruction.Instruction: Instruction of the Oracle.

    """
    # get shape of training data
    train_shape = np.shape(train_data)
    # check if training data is two dimensional
    if len(train_shape) != 2:
        raise ValueError("Provided training data not 2-dimensional. Provide"
                         "a matrix of shape n_samples x dim")
    # get the log2 values of the number of samples
    #  m from training data and dimension of vectors n
    m, n = np.log2(train_shape)
    # check if m should be ceiled.
    if not m.is_integer():
        warnings.warn("Number of training states not a positive power of 2,"
                      "adding extra qubit to comply.")
        m = np.ceil(m)

    # create quantum registers
    r_train = qk.QuantumRegister(n, name='train_states')
    r_comp_basis = qk.QuantumRegister(m, name='comp_basis')

    # initialize the list containing the controlled inits, which will assign
    #  each training state i to r_train (so should be as long as number of
    #  samples)
    controlled_inits = [qcirc.ControlledGate] * train_shape[0]

    # create an empty circuit with the registers
    oracle_circ = qk.QuantumCircuit(
        r_train,
        r_comp_basis,
        name='oracle'
    )

    # create all the controlled inits for each vector in train data
    for i, train_state in enumerate(train_data):
        controlled_inits[i] = \
            gates.controlled_initialize(
                r_train,
                train_state,
                num_ctrl_qubits=r_comp_basis.size,
                name="phi_{}".format(i)
            )

    # apply all the x-gates and controlled inits to the circuit
    # define the length of the binary string which will translate |i>
    bin_number_length = r_comp_basis.size
    where_x = where_to_apply_x(bin_number_length)

    for i, (c_init, x_idx) in enumerate(zip(controlled_inits, where_x)):
        # represent i in binary number with length bin_number_length, so
        #  leading zeros are included
        logger.debug(f"applying x-gates to: {x_idx}")
        # apply the x-gates
        oracle_circ.x(r_comp_basis[x_idx])
        # apply the controlled init
        oracle_circ.append(c_init, r_comp_basis[:] + r_train[:])

    logger.debug(f"Created oracle as:\n{oracle_circ.draw()}")

    return oracle_circ.to_instruction()


def where_to_apply_x(bin_number_length: int) -> List:
    r""" Create an array to apply :math:`X`-gates systematically to create all
    possible register combinations.

    This method returns the indices on where to apply :math:`X`-gates on a
    quantum register with ``n`` qubits to generate all possible binary numbers
    on that register.

    Example:
        Suppose we have a register with 2 qubits. We want to make sure we check
        all possible states this register can be in, such that a data point
        can be assigned. A register with 2 qubits can be in 4 states:

        .. math::

            |0\rangle = |00\rangle, |1\rangle = |01\rangle,
            |2\rangle = |10\rangle, |3\rangle = |11\rangle

        So to apply :math:`\phi_1`, the register must be in state
        :math:`|01\rangle`, and we need to apply the :math:`X`-gate only to the
        first qubit. The state becomes :math:`|11\rangle` and the controlled
        initialise will trigger.

        Because the algorithm will check for all states in succession, this can
        be reduced to prevent double placements of :math:`X`-gates, and it
        determines where to place the :math:`X`-gates via:

        .. math:: |i-1\rangle XOR |i\rangle

        A full list of all these configurations is created by this method\:

        .. jupyter-execute::

            from qiskit_quantum_knn.qknn.qknn_construction import where_to_apply_x

            num_qubits = 2
            where_to_apply_x(num_qubits)

    Args:
        bin_number_length (int): the length of the highest binary value (or
            the number of qubits).

    Returns:
        List: All possible combinations.

            A length ``2**bin_number_length`` of the indices of the qubits where
            the :math:`X`-gate must be applied to.
    """
    powers_of_two = 2 ** np.arange(bin_number_length)
    indices = \
        [
            [
                ind for ind, v in enumerate(powers_of_two)
                if v & (pos ^ (pos - 1)) == v
            ] for pos in range(2 ** bin_number_length)
        ]
    return indices


def initialise_qknn(log2_dim: int,
                    log2_n_samps: int,
                    test_state: np.ndarray) -> qk.QuantumCircuit:
    r"""Creates the registers and applies the unclassified datum :math:`\psi`.

    Coincides with Step 1: the "initialisation" section in
    :afham2020:`Afham et al. (2020)`. Initialises a QuantumCircuit
    with 1 + 2n + m qubits (n: log2_dimension, m: log2_samples) for a QkNN
    network, where qubits 1 till n are initialised in some state psi (
    state_to_classify).

    Example:
        Set up the scaffolds for a QkNN :class:`QuantumCircuit`.

        .. jupyter-execute::

            from qiskit_quantum_knn.qknn.qknn_construction import initialise_qknn

            n_dim_qubits = 1
            n_samps_qubits = 1
            test_state = [0, 1]

            init_circ = initialise_qknn(n_dim_qubits, n_samps_qubits, test_state)
            print(init_circ.draw())

    Args:
        log2_dim (int): int, log2 value of the
            dimension of the test and train states.
        log2_n_samps (int): int,
            log2 value of the number of training samples M.
        test_state (numpy.ndarray): 2 ** log2_dimension complex values to
            initialise the r_1 test state in (psi).

    Returns:
        QuantumCircuit: The initialised circuit.
    """
    if len(test_state) != 2 ** log2_dim:
        raise ValueError(
            "Dimensionality of test state or provided dimension not correct;"
            " test state dim is {0:d}, and dimension given is {1:d}".format(
                len(test_state), 2 ** log2_dim
            )
        )

    # register for control qubit
    r_0 = qk.QuantumRegister(1, name='control')
    # register for test state
    r_1 = qk.QuantumRegister(log2_dim, name='state_to_classify')
    # register for train state
    r_2 = qk.QuantumRegister(log2_dim, name='train_states')
    # register for computational basis
    r_3 = qk.QuantumRegister(log2_n_samps, name='comp_basis')

    # classical register for measuring the control and computational basis
    c_0 = qk.ClassicalRegister(r_0.size, name='meas_control')
    c_1 = qk.ClassicalRegister(r_3.size, name="meas_comp_basis")

    init_circ = qk.QuantumCircuit(r_0, r_1, r_2, r_3, c_0, c_1)
    init = qi.Isometry(test_state, 0, 0)
    init.name = "init test state"
    init_circ.append(init, r_1)
    init_circ.barrier()

    logger.debug(f"Initialised circuit as:\n{init_circ.draw()}")

    return init_circ


def state_transformation(qknn_circ: qk.QuantumCircuit,
                         oracle: qinst.Instruction) -> qk.QuantumCircuit:
    r"""applies :math:`H`-gates and the Oracle :math:`\mathcal{W}` to the
    circuit, and applies the :math:`SWAP`-test.

    Coincides with Step 2: the "state transformation" section from
    :afham2020:`Afham et al. (2020)`. Applies Hadamard gates and
    Quantum Oracle to bring :math:`r_1, r_2, r_3, r_4` in the desired states.

    Note:
        This needs the :class:`QuantumCircuit` created by
        :py:func:`initialise_qknn` as a parameter in order to function
        properly.

    Example:
        Apply the oracle and test data in a :class:`QuantumCircuit`.

        .. jupyter-execute::

            from qiskit_quantum_knn.qknn.qknn_construction import create_oracle, \
                initialise_qknn, state_transformation

            n_dim_qubits = 1  # must be log(len(test_state))
            n_samps_qubits = 1  # must be log(len(train_data))

            test_state = [0, 1]
            train_data = [
                [1, 0],
                [0, 1]
            ]

            oracle = create_oracle(train_data)

            init_circ = initialise_qknn(n_dim_qubits, n_samps_qubits, test_state)
            state_circ = state_transformation(init_circ, oracle)
            print(state_circ.draw())

    Args:
        qknn_circ (QuantumCircuit): has been initialised according to
            initialise_qknn().
        oracle (qiskit Instruction): oracle W|i>|0> = W|i>|phi_i> for applying
            training data.

    Returns:
        QuantumCircuit: the transformed :class:`QuantumCircuit`.

    """
    # initialising registers for readability
    [control, test_register, train_register, comp_basis] = qknn_circ.qregs

    # perform equation 13 from Afham; Afham, Afrad; Goyal, Sandeep K. (2020).
    qknn_circ.h(control)
    qknn_circ.h(comp_basis)

    # perform equation 15
    # append to circuit
    qknn_circ.append(oracle, train_register[:] + comp_basis[:])

    # controlled swap
    for psi_bit, phi_bit in zip(test_register, train_register):
        qknn_circ.cswap(control, psi_bit, phi_bit)

    # final Hadamard gate
    qknn_circ.h(control)

    # barrier to round it of
    qknn_circ.barrier()

    logger.info(f"transformed registers to circuit:\n{qknn_circ.draw()}")

    return qknn_circ


def add_measurements(qknn_circ: qk.QuantumCircuit) -> qk.QuantumCircuit:
    """Adds measurement gates to the control and computational basis.

    Performs the third and final step of the building of the QkNN circuit by
    adding measurements to the control qubit and the computational basis.

    Note:
        This needs the :class:`QuantumCircuit` created by
        :py:func:`state_transformation` as a parameter in order to function
        properly.

    Example:

        .. jupyter-execute::

            from qiskit_quantum_knn.qknn.qknn_construction import create_oracle, \
                initialise_qknn, state_transformation, add_measurements

            n_dim_qubits = 1  # must be log(len(test_state))
            n_samps_qubits = 1  # must be log(len(train_data))

            test_state = [0, 1]
            train_data = [
                [1, 0],
                [0, 1]
            ]

            oracle = create_oracle(train_data)

            init_circ = initialise_qknn(n_dim_qubits, n_samps_qubits, test_state)
            state_circ = state_transformation(init_circ, oracle)
            final_circ = add_measurements(state_circ)
            print(final_circ.draw())

    Args:
        qknn_circ (qk.QuantumCircuit): has been build up by first applying
                                       initialise_qknn() and
                                       state_transformation().

    Returns:
        QuantumCircuit: the :class:`QuantumCircuit` with measurements applied.
    """
    comp_basis_creg = qknn_circ.cregs[-1]
    comp_basis_qreg = qknn_circ.qregs[-1]
    qknn_circ.measure(qknn_circ.qregs[0], qknn_circ.cregs[0])
    for qbit, cbit in zip(comp_basis_qreg, reversed(comp_basis_creg)):
        qknn_circ.measure(qbit, cbit)

    logger.debug("Added measurements.")

    return qknn_circ
