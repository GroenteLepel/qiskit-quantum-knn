# Qiskit Quantum kNN


| [![INGLogo](http://logok.org/wp-content/uploads/2014/11/ING_logo.png)][ing-home]  | [![RULogo](https://www.ru.nl/publish/pages/954125/ru_en_1.jpg)][ru-home] |
:---:|:---:

[![License](https://img.shields.io/github/license/GroenteLepel/qiskit-quantum-knn)](https://opensource.org/licenses/Apache-2.0)
[![](https://img.shields.io/github/v/release/GroenteLepel/qiskit-quantum-knn)](https://github.com/GroenteLepel/qiskit-quantum-knn/releases/)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FGroenteLepel%2Fqiskit-quantum-knn%2Fbadge%3Fref%3Dmaster&style=flat)](https://actions-badge.atrox.dev/GroenteLepel/qiskit-quantum-knn/goto?ref=master)
[![Documentation Status](https://readthedocs.org/projects/qiskit-quantum-knn/badge/?version=latest)](https://qiskit-quantum-knn.readthedocs.io/en/latest/?badge=latest)


**Qiskit Quantum kNN** is a pure quantum knn classifier for a gated quantum
computer, which is build with [**Qiskit**][qiskit-github].

Qiskit Quantum kNN is made as a final project to fulfill a master's degree
at the Radboud University Nijmegen, in collaboration with ING Quantum 
Technology. It is build by using [Afham et al. (2020)][afham2020] as it's
primary guide on how to construct the quantum circuit used for distance
measurements.

## Installation
The best way of installing `qiskit-quantum-knn` is by using `pip`:

```bash
$ pip install qiskit-quantum-knn
```

Since `qiskit-quantum-knn` runs mainly by using `qiskit`, it is advised to check
out their [installation guide][3] on how to install Qiskit.

## License
[Apache License 2.0](LICENSE.txt)

[ing-home]: https://www.ing.nl/particulier/english/index.html "ING business home"
[ru-home]: https://www.ru.nl/ "RU homepage"
[qiskit-github]: https://github.com/Qiskit/qiskit
[afham2020]: https://arxiv.org/abs/2003.09187 "Quantum k-nearest neighbor machine learning algorithm"
[3]: https://qiskit.org/documentation/install.html