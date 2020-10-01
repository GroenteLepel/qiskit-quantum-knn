import logging
from typing import List

import numpy as np
import qiskit as qk
import qiskit.extensions.quantum_initializer as qi
import qiskit.circuit.instruction as qinst

import qiskit_quantum_knn.qknn.quantumgates as gates
from qiskit_quantum_knn.config import Config

conf = Config()
logger = logging.getLogger(conf["LOGGER"])

"""Construction of a qknn QuantumCircuit."""


def construct_circuit(state_to_classify: np.ndarray,
                      training_data: np.ndarray,
                      add_measurement: bool) -> qk.QuantumCircuit:
    """Setup for a qknn QuantumCircuit.
    Constructs the qknn QuantumCircuit according to the stepwise "instructions"
    in Afham; Basheer, Afrad; Goyal, Sandeep K. (2020).
    Args:
        state_to_classify (numpy.ndarray): array of dimension N complex
            values describing the state to classify via KNN.
        training_data (numpy.ndarray): array containing M training samples
            of dimension N.
        add_measurement (bool): controls if measurements must be added
            to the classical registers.
    Raises:
        ValueError: If the length of the vectors in the classified_states and/or
        test data are not a positive power of 2.
    Returns:
        QuantumCircuit: constructed circuit.
    """

    state_dimension = len(state_to_classify)
    n_train_samples = len(training_data)

    n, m = np.log2(state_dimension), np.log2(n_train_samples)

    # Check if param is a power of 2
    if (n == 0 or not n.is_integer()) and (m == 0 or not m.is_integer()):
        raise ValueError(
            "Desired statevector length not a positive power of 2."
        )

    # step 1
    qknn_circ = initialise_qknn(n, m, state_to_classify)
    # step 2
    qknn_circ = state_transformation(qknn_circ, training_data)
    # step 3
    if add_measurement:
        qknn_circ = add_measurements(qknn_circ)

    logger.debug(f"Final circuit:\n{qknn_circ.draw(fold=90)}")

    return qknn_circ


def initialise_qknn(log2_dim: int,
                    log2_n_samps: int,
                    test_state: np.ndarray) -> qk.QuantumCircuit:
    """
    Coincides with Step 1: the "initialisation" section in Afham;
    Basheer, Afrad; Goyal, Sandeep K. (2020). Initialises a QuantumCircuit
    with 1 + 2n + m qubits (n: log2_dimension, m: log2_samples)
    for a qknn network, where qubits 1 till n are initialised in some state
    psi (state_to_classify).
    Args:
        log2_dim (int): int, log2 value of the dimension of the test and
            train states.
        log2_n_samps (int): int, log2 value of the number of training samples M.
        test_state (numpy.ndarray): 2 ** log2_dimension complex values to
            initialise the r_1 test state in (psi).
    Returns:
        QuantumCircuit: initialised circuit.
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
                         train_data: np.ndarray) -> qk.QuantumCircuit:
    """
    Coincides with Step 2: the "state transformation" section from Afham;
    Basheer, Afrad; Goyal, Sandeep K. (2020). Applies Hadamard gates and
    Quantum Oracle to bring r_1, r_2, r_3 and r_4 in the desired states.
    Args:
        qknn_circ (QuantumCircuit): has been initialised according to
            initialise_qknn().
        train_data (List): List with n_train_samples complex vectors of
            dimension state_dimension.
    Returns:
        QuantumCircuit: the QuantumCircuit with state transformation applied.
    """
    # initialising registers for readability
    [control, test_register, train_register, comp_basis] = qknn_circ.qregs

    # perform equation 13 from Afham; Basheer, Afrad; Goyal, Sandeep K. (2020).
    qknn_circ.h(control)
    qknn_circ.h(comp_basis)

    # perform equation 15
    # create oracle
    oracle = create_oracle(train_register, comp_basis, train_data)
    # append to circuit
    qknn_circ.append(oracle, train_register[:] + comp_basis[:])

    # controlled swap
    for psi_bit, phi_bit in zip(test_register, train_register):
        qknn_circ.cswap(control, psi_bit, phi_bit)

    # final Hadamard gate
    qknn_circ.h(control)

    # barrier to round it of
    qknn_circ.barrier()

    logger.debug(f"transformed registers to circuit:\n{qknn_circ.draw()}")

    return qknn_circ


# noinspection PyTypeChecker
def create_oracle(r_train: qk.QuantumRegister,
                  r_comp_basis: qk.QuantumRegister,
                  train_data) -> qinst.Instruction:
    """
    Creates an oralce to perform as:
        W |0>|basis_state> = |phi>|basis_state>,     (14)
    where the equation number refers to that of Afham; Basheer, Afrad; Goyal,
    Sandeep K. (2020). Creates oracle to bring qubit into desired state |phi>
    as Instruction, this can be appended to the desired circuit.
    Args:
        r_train (QuantumRegister): initialized to the train data.
        r_comp_basis (QuantumRegister): contains the computational
                                        basis |i> to bring r_train into state
                                        classified_states[i]
        train_data (List): List of vectors with dimension len(r_train) to
                           initialize r_train to.
    Returns:
        circuit.instruction.Instruction: Instruction of the Oracle.
    """
    controlled_inits = [qinst.Instruction] * len(train_data)

    # initialize the circuit
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

    bin_number_length = r_comp_basis.size
    # "kick start" the loop by defining the first "previous" number
    prev_bin_array = np.ones(bin_number_length, dtype=int)

    x_location_values = np.arange(r_comp_basis.size) + 1
    for i, c_init in enumerate(controlled_inits):
        # represent i in binary number with length bin_number_length, so
        #  leading zeros are included
        binary_string = "{1:0{0:d}b}".format(bin_number_length, i)
        # convert this to array
        curr_bin_array = np.array(list(binary_string), dtype=int)
        indices = np.logical_xor(prev_bin_array, curr_bin_array)
        where_to_apply_x = np.arange(bin_number_length)[indices].tolist()
        oracle_circ.x(r_comp_basis[where_to_apply_x])
        oracle_circ.append(c_init, r_comp_basis[:] + r_train[:])

        prev_bin_array = curr_bin_array

    logger.debug(f"Created oracle as:\n{oracle_circ.draw()}")

    return oracle_circ.to_instruction()


def add_measurements(qknn_circ: qk.QuantumCircuit) -> qk.QuantumCircuit:
    """
    Performs the third and final step of the building of the qknn circuit by
    adding measurements to the control qubit and the computational basis.
    Args:
        qknn_circ (qk.QuantumCircuit): has been build up by first applying
                                       initialise_qknn() and
                                       state_transformation().
    Returns:
        QuantumCircuit: the qknn_circ with measurements applied.
    """
    comp_basis_creg = qknn_circ.cregs[-1]
    comp_basis_qreg = qknn_circ.qregs[-1]
    qknn_circ.measure(qknn_circ.qregs[0], qknn_circ.cregs[0])
    for qbit, cbit in zip(comp_basis_qreg, reversed(comp_basis_creg)):
        qknn_circ.measure(qbit, cbit)

    logger.debug("Added measurements.")

    return qknn_circ
