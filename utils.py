from itertools import dropwhile

import numpy as np


def count_consecutive_ones_from_end(lst):
    reversed_lst = reversed(lst)
    count = len(list(dropwhile(lambda x: x == 1, reversed_lst)))
    return len(lst) - count


def concat_multiple_times(array1, array2, times):
    repeated_2 = np.repeat(array2, times, axis=0)
    return np.concatenate([array1, repeated_2])
