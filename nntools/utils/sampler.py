import numpy as np


def sample(sampling_values):
    """
    Return a numeric sample
    :param sampling_values:
        - if list: return a value randomly picked from the interval in the list following a uniform distribution
        - if tuple: Randomly return one of the list
    :return:
    """
    if np.isscalar(sampling_values):
        return sampling_values
    if sampling_values[0] == sampling_values[1]:
        return sampling_values[0]
    if isinstance(sampling_values, list):
        return np.random.uniform(sampling_values[0], sampling_values[1])
    if isinstance(sampling_values, tuple):
        return np.random.choice(sampling_values)
