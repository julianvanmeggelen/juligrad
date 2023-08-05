from juligrad.tensor import Tensor

import os
from enum import Enum
from urllib.request import urlretrieve
from urllib.parse import urljoin
import gzip
import functools
import operator
import array
import numpy
import struct
import numpy as np

DATA_DIR = './data/'
if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)


# *** MNIST ***
class MNIST(Enum):
    train = ('train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz')
    test = ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


def MNIST_parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.

    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse

    endian : str
        Byte order of the IDX file. See [1] for available options

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file

    1. https://docs.python.org/3/library/struct.html
        #byte-order-size-and-alignment

    https://github.com/datapythonista/mnist/blob/master/mnist/__init__.py
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise ValueError('Invalid IDX file, '
                             'file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise ValueError('Invalid IDX file, '
                             'file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise ValueError('Unknown data type '
                             '0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise ValueError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items,
                                                          len(data)))

    return numpy.array(data).reshape(dimension_sizes)

def convertOneHot(a: Tensor):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def load_mnist(set:MNIST = MNIST.train, dir=None, limit:int=None):
    if dir is None: dir = os.path.join(DATA_DIR,'mnist')
    URL = 'http://yann.lecun.com/exdb/mnist/'
    def _load(dfile: str):
        fname = os.path.join(DATA_DIR, dfile)
        if not os.path.isfile(fname):
            url = urljoin(URL, dfile)
            urlretrieve(url, fname)
        fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
        with fopen(fname,'rb') as fd:
            res = MNIST_parse_idx(fd)
            return res[:limit] if limit is not None else res
 
    return Tensor.fromNumpy(_load(set.value[0])[:,np.newaxis,:,:], requiresGrad=False), Tensor.fromNumpy(convertOneHot(_load(set.value[1])), requiresGrad=False)


