# Qiskit Quantum kNN


| <img src="http://logok.org/wp-content/uploads/2014/11/ING_logo.png" width="1000">  | <img src="https://www.ru.nl/views/ru-baseline/images/logos/ru_nl.svg" width="1000"> |
:---:|:---:


[![License](https://img.shields.io/github/license/GroenteLepel/qiskit-quantum-knn)](https://opensource.org/licenses/Apache-2.0)
[![](https://img.shields.io/github/v/release/GroenteLepel/qiskit-quantum-knn)](https://github.com/GroenteLepel/qiskit-quantum-knn/releases/)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FGroenteLepel%2Fqiskit-quantum-knn%2Fbadge%3Fref%3Dmaster&style=flat)](https://actions-badge.atrox.dev/GroenteLepel/qiskit-quantum-knn/goto?ref=master)
[![Documentation Status](https://readthedocs.org/projects/qiskit-quantum-knn/badge/?version=latest)](https://qiskit-quantum-knn.readthedocs.io/en/latest/?badge=latest)

> [!NOTE]
> This project is archived on the 14th of march, 2025. It was way overdue.
> I wrote this code at the time that Qiskit was still split in multiple
> modules, but I briefly came past it at a time and saw that everything
> is different now. This code does not work anymore in that way, and even
> though some great enthusiastic people helped me keep it up-to-date, the
> repo fell into neglect. I'm no longer working in this field of work,
> but I am still proud of this accomplishment. I will keep this here
> because of that, and because of the potential value that it could
> probably deliver to someone else.

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
