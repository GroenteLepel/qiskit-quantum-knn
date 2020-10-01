from typing import Optional

import numpy as np
import qiskit as qk
import qiskit.extensions.quantum_initializer as qi
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate

"""
Extra file containing some decomposed quantum gates to get familiar with them 
and Qiskit.
"""


def swap():
    """
    Self-written decomposed version of swap gate:
    q_1: ───X─
            │
    q_2: ───X─
    :return: instruction of this gate
    """
    swap_circuit = qk.QuantumCircuit(2, name='swap_gate')
    swap_circuit.cx(0, 1)
    swap_circuit.cx(1, 0)
    swap_circuit.cx(0, 1)
    return swap_circuit.to_instruction()


def fidelity_instruction():
    """
    Fidelity is P(0) - P(1)
         ┌───┐   ┌───┐┌─┐
    q_0: ┤ H ├─■─┤ H ├┤M├
         └───┘ │ └───┘└╥┘
    q_1: ──────X───────╫─
               │       ║
    q_2: ──────X───────╫─
                       ║
    c_0: ══════════════╩═
    :return: instruction of the gate
    """
    fidelity_circ = qk.QuantumCircuit(3, 1)
    fidelity_circ.h(0)
    fidelity_circ.cswap(0, 1, 2)
    fidelity_circ.h(0)
    fidelity_circ.measure(0, 0)

    fidelity_instr = fidelity_circ.to_instruction()

    return fidelity_instr


def init_to_state(reg_to_init: qk.QuantumRegister,
                  init_state: np.ndarray,
                  name: Optional[str] = None) -> Gate:
    """Initialize a regester to provided state.
    Args:
        reg_to_init (QuantumRegister): register which needs to be initialized.
        init_state (np.ndarray): state to which the reg_to_init must be
            initialized to.
        name (str): optional, name for the init_gate.
    Raises:
        ValueError: if the register and state do not have the same dimension.

    Returns:
        Gate: gate of size reg_to_init.size which performs the initialization.
    """
    # check if provided values are correct
    if len(init_state) != 2 ** len(reg_to_init):
        raise ValueError(
            "Dimensionality of the init_state does not coincide with the "
            "length of the register to initialise to: is {0} and {1}".format(
                len(init_state), len(reg_to_init)
            )
        )

    init_circ = qk.QuantumCircuit(reg_to_init, name=name)  # create temp circuit
    init = qi.Isometry(init_state, 0, 0)  # create Isometry for init
    init_circ.append(init, reg_to_init)  # apply init to temp circuit

    basis_gates = ['u1', 'u2', 'u3', 'cx']  # list basis gates
    # transpile circuit so that it is decomposed to the basis gates above,
    #  making it unitary and possible to convert from Instruction to Gate
    transpiled = qk.transpile(init_circ, basis_gates=basis_gates)
    init_gate = transpiled.to_gate()  # convert to Gate
    return init_gate


def controlled_initialize(reg_to_init: qk.QuantumRegister,
                          init_state: np.ndarray,
                          num_ctrl_qubits: Optional[int] = 1,
                          name: Optional[str] = None) -> ControlledGate:
    """Initialize a register to provided state with control.
    Args:
        reg_to_init (QuantumRegister): register which needs to be initialized.
        init_state (np.ndarray): state to which the reg_to_init must be
            initialized to.
        num_ctrl_qubits (int): optional, number of desired controls.
        name (str): optional, name for the init_gate.
    Returns:
        ControlledGate: gate of size reg_to_init.size + num_ctrl_qubits which
            performs the initialize with control.
    """
    # create the init state
    init_gate = init_to_state(reg_to_init, init_state, name)
    # make it controlled
    controlled_init = init_gate.control(num_ctrl_qubits=num_ctrl_qubits)

    return controlled_init
