import numpy as np
import copy



def make_noisy_general(clean_data, noise_matrix, random_state, num_classes):
    """ Perturbs the MNIST labels based on the probabilities of the given noise matrix

    Args:
        clean_data: list of instances
        noise_matrix: defines the noise process
        random_state: for reproducibility

    Returns:
        A perturbed copy of clean_data (the noisy_data)
    """

    # assert that the sum of a row is close to 1 (within a tolerance)
    for row in noise_matrix:
        assert np.isclose(np.sum(row), 1)
    # print(f"noise matrix is {noise_matrix.shape}")
    # length of noise matrix is equal to the number of classes 
    assert len(noise_matrix) == num_classes
    # make a deep copy of the clean data to corrupt
    noisy_data = copy.deepcopy(clean_data)
    # print(f"clean data shape is {clean_data.shape}")
    # iterate through the noisy data (one item at a time)
    for i in range(len(noisy_data)):
        # print(f"noisy data at i is {noisy_data[i]}")
        # is this right, ask Dr. Hanan?
        probability_row = noise_matrix[noisy_data[i]]
        # print(probability_row)
        # chose randomly from the classes, the probability row provides probabilies for each of the 4 values. 
        noisy_data[i] = random_state.choice(num_classes, p=probability_row)
        # labels are now corrupted 
    # return the noisy data 
    return noisy_data

# inject specific type of noise according to noise level in the dataset 
def make_data_noisy(y, noise_level, noise_type, r_state, num_classes):
    assert noise_type in ['sflip', 'uniform', 'uniform_m']

    if noise_type == 'sflip':
        _, noisy_data = make_noisy_single_flip(y, noise_level, r_state, num_classes)
    elif noise_type == 'uniform':
        _, noisy_data = make_noisy_uniform(y, noise_level, r_state, num_classes)
    elif noise_type == 'uniform_m':
        _, noisy_data = make_noisy_uniform_m(y, noise_level, r_state, num_classes)
    else:
        raise NotImplementedError('noise type not supported')


    return noisy_data

def make_noisy_uniform(y, noise_level, r_state, num_classes):
    """
    Args: 
    y: clean labels
    noise_level: used to generate the noise matrix 
    r_state: random state for reproducibility 

    returns:
    the noise matrix, and the corrupted labels
    """
    assert num_classes == len(set(y))
    # noise level is a probability value 
    clean_label_probability = 1 - noise_level
    # noise level is equally distributed across all the labels (even clean label probability) ?
    uniform_noise_probability = noise_level / num_classes  # distribute noise_level across all other labels
    clean_label_probability += uniform_noise_probability

    # initialize empty matrix of shape num_classes x num_classes 
    true_noise_matrix = np.empty((num_classes, num_classes))
    # fill the array with a scalar value: in this case the uniform_noise_probability
    true_noise_matrix.fill(uniform_noise_probability)
    # true noise matrix is a square matrix where the principal diagonal is equal to clean_label_probability , 
    # and the rest is equal to uniform_noise_probability
    for true_label in range(num_classes):
        true_noise_matrix[true_label][true_label] = clean_label_probability

    noisy_data = make_noisy_general(y, true_noise_matrix, r_state, num_classes)

    return true_noise_matrix, noisy_data

def get_uniform_m_flip_mat(noise_level, num_classes):
    clean_label_probability = 1 - noise_level
    uniform_noise_probability = noise_level / (num_classes - 1)  # distribute noise_level across all other labels

    true_noise_matrix = np.empty((num_classes, num_classes))
    true_noise_matrix.fill(uniform_noise_probability)
    for true_label in range(num_classes):
        true_noise_matrix[true_label][true_label] = clean_label_probability

    return true_noise_matrix


def make_noisy_uniform_m(y, noise_level, r_state, num_classes):
    assert num_classes == len(set(y))

    true_noise_matrix = get_uniform_m_flip_mat(noise_level, num_classes)

    noisy_data = make_noisy_general(y, true_noise_matrix, r_state, num_classes)

    return true_noise_matrix, noisy_data



def get_single_flip_mat(noise_level, num_classes):
    flips = np.arange(num_classes)
    flips = np.roll(flips, 1)

    true_noise_matrix = np.zeros((num_classes, num_classes))
    for true_label in range(num_classes):
        true_noise_matrix[true_label][true_label] = 1 - noise_level
        true_noise_matrix[true_label][flips[true_label]] = noise_level
    return true_noise_matrix



def make_noisy_single_flip(y, noise_level, r_state, num_classes):
    assert num_classes == len(set(y))
    true_noise_matrix = get_single_flip_mat(noise_level, num_classes)

    noisy_data = make_noisy_general(y, true_noise_matrix, r_state, num_classes)

    return true_noise_matrix, noisy_data
