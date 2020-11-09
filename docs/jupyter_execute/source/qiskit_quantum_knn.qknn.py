#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from qiskit_quantum_knn.qknn.qknn_construction import create_qknn

test_data = [1, 0]

train_data = [
    [1, 0],
    [1, 0]
]

circuit = create_qknn(test_data, train_data, add_measurement=True)
print(circuit.draw())

