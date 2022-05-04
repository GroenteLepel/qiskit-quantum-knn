"""Abstract base class for the binary classifier and the multiclass classifier."""

from abc import ABC, abstractmethod


class _QSVM_ABC(ABC):
    """Abstract base class for the binary classifier and the multiclass classifier."""

    def __init__(self, qalgo):

        self._qalgo = qalgo
        self._ret = {}

    @abstractmethod
    def run(self):
        """ run """
        raise NotImplementedError("Must have implemented this.")

    @property
    def ret(self):
        """ return result """
        return self._ret

    @ret.setter
    def ret(self, new_ret):
        """ sets result """
        self._ret = new_ret
