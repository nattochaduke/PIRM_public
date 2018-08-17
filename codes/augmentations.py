import numpy as np

def random_flip(*arrays):
    if np.random.choice([True, False]):
        return (np.flip(arr, axis=2) for arr in arrays)
    else:
        return arrays

def random_rotation(*arrays):
    k = np.random.randint(low=0, high=4)
    return (np.rot90(arr, k, axes=(1, 2)) for arr in arrays)

def cutout(patchsize, scale, probability=0.3):
    """
    Zeros random square region of input image.
    :param patchsize:
    :param scale:
    :param probability:
    :return: operator function.
    """
    def operator(array1, array2):
        if np.random.choice([True, False], p=(probability, 1-probability)):
            size = np.random.randint(low=5, high=patchsize // 2)
            uppermost = np.random.randint(low=0, high=patchsize-size)
            leftmost = np.random.randint(low=0, high=patchsize-size)
            return (array1[uppermost: uppermost+size][leftmost: leftmost+size],
                    array2[scale*uppermost: scale*(uppermost+size)][scale*leftmost: scale*(leftmost+size)])

        else:
            return (array1, array2)
    return operator
