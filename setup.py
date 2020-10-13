import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qiskit_quantum_knn",
    version="1.0.0",
    author="Daniël J. Kok",
    author_email="djonatankok@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GroenteLepel/qiskit-quantum-knn",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "qiskit"
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
    python_requires='>=3.6',
    extras_require={
        'visualization': ['matplotlib>=2.1', 'ipywidgets>=7.3.0',
                          'pydot', "pillow>=4.2.1", "pylatexenc>=1.4",
                          "seaborn>=0.9.0", "pygments>=2.4"],
    }
)
