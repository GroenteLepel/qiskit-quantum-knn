Qiskit Quantum kNN: A pure quantum knn classifier for a gated quantum computer.
==================
Welcome to the documentation page for ``qiskit-quantum-knn``. This project
is created to complete the Masters course "Particle- and Astrophysics" at
the Radboud University in Nijmegen. Hence, this project is made by a single
person, which means that there might be issues. If you should find any issues,
please report these at the
`GitHub page <https://github.com/GroenteLepel/qiskit-quantum-knn>`_.

Requirements
------------
The following package is required for this to work:
 * Qiskit (found `here <https://qiskit.org/>`_)

Dependencies
------------
The same dependencies needed for Qiskit are also needed for this package to
work.

Usage
-----
Once installed, ``qiskit-quantum-knn`` can be used as follows:

.. jupyter-execute::

    import qiskit_quantum_knn

A small example on how to use this for classification:

.. jupyter-execute::

    from qiskit_quantum_knn.qknn import QKNeighborsClassifier
    from qiskit_quantum_knn.encoding import analog
    from qiskit.utils import QuantumInstance
    from sklearn import datasets
    import qiskit as qk

    # initialising the quantum instance
    backend = qk.BasicAer.get_backend('qasm_simulator')
    instance = QuantumInstance(backend, shots=10000)

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

More info on how to work with the Quantum kNN is explained thoroughly in
the rest of the documentation.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   source/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
