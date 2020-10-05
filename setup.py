import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qiskit_quantum_knn",
    version="0.0.2",
    author="Daniël J. Kok",
    author_email="djonatankok@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GroenteLepel/qiskit-quantum-knn",
    packages=setuptools.find_packages(),
    install_requirest=[
        "qiskit",
        "numpy",
        "scipy"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.6'
)
