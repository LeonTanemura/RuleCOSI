# This file is derived from "bitarray_backend.py" from the repository:
# GitHub: https://github.com/jobregon1212/rulecosi/blob/9928a4a586ec90f35021df80d01c1f5347b037ab/rulecosi/bitarray_backend.py
# Original Author: jobregon1212
# Description: Portions of code have been copied to implement bitarray backend handling.

# Note: The original code is available at https://github.com/jobregon1212/rulecosi

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from gmpy2 import popcount
from sys import byteorder

from bitarray import bitarray


class BitarrayBackend(metaclass=ABCMeta):
    """ Backend class for handling different libraries used for the array of
    bits representations and their related computations.

    """

    def __init__(self, n, classes):
        self.n = n
        self.classes = classes
        self.n_classes = len(classes)

    @abstractmethod
    def generate_ones(self):
        """

        :return:
        """

    @abstractmethod
    def generate_zeros(self):
        """

        :return:
        """

    @abstractmethod
    def get_length(self, array):
        """

        :param array:
        :return:
        """

    @abstractmethod
    def get_number_ones(self, array):
        """

        :param array:
        :return:
        """

    @abstractmethod
    def get_complement(self, array, ones=None):
        """

        :param ones:
        :param array:
        :return:
        """


    @abstractmethod
    def get_array(self, bool_array):
        """

        :param array:
        :return:
        """


class PythonIntArray(BitarrayBackend):
    def generate_ones(self):
        return int.from_bytes(
            np.packbits(np.ones(self.n, dtype=bool)),
            byteorder=byteorder)

    def generate_zeros(self):
        return int.from_bytes(
            np.packbits(np.zeros(self.n, dtype=bool)),
            byteorder=byteorder)

    def get_length(self, array):
        return array.bit_length()

    def get_number_ones(self, array):
        return popcount(array)

    def get_complement(self, array, ones=None):
        return array ^ ones

    def get_array(self, bool_array):
        return int.from_bytes(np.packbits(bool_array),
                              byteorder=byteorder)


class BitArray(BitarrayBackend):
    def generate_ones(self):
        b_array = bitarray(self.n)
        b_array.setall(True)
        return b_array

    def generate_zeros(self):
        b_array = bitarray(self.n)
        b_array.setall(False)
        return b_array

    def get_length(self, array):
        return len(array)

    def get_number_ones(self, array):
        return array.count(1)

    def get_complement(self, array, ones=None):
        return ~array

    def get_array(self, bool_array):
        return bitarray(bool_array.astype(int).tolist())