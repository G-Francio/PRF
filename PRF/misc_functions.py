import numpy as np
from numba import jit

############################################################
############################################################
# Modified script from here
from scipy.integrate import quad
from scipy.stats import norm

K = 10

# Precompute the distribution, extrema go from 0 to MAX_LM
MAX_LM = K
X_LM = np.arange(0, MAX_LM, 0.01)


# build cumulative distribution
def LMDist(x):
    return 1 / (K + np.exp(-K) - 1) * (1 - np.exp(-x))


def LMCumDist(u):
    return quad(LMDist, 0, u)[0]


# Trasformo nella versione vettorizzata
vectLMCumDist = np.vectorize(LMCumDist)

# Calcolo sui valori tabulati
LM_VALUES = vectLMCumDist(X_LM)
LM_VALUES = np.append(LM_VALUES, 1)

print("K Value: {}".format(K))
print("Considering upper limits")

############################################################
############################################################
############ Propogate Probabilities functions  ############
############################################################
############################################################

# grid is finer than original one
N_SIGMA = 3
X_GAUS = np.arange(-N_SIGMA, N_SIGMA, 0.01)
GAUS = np.array(norm(0, 1).cdf(X_GAUS))
GAUS = np.append(GAUS, 1)


# TODO: This function needs some clean up and a _check_flag somewhere
#  to preprocess things, it is such a mess
@jit
def split_probability(value, delta, flag, threshold):
    """
    Calculate split probability for a single object
    """
    # flag != 0 -> upper limit (carries info for turnover)

    if np.isnan(value):
        return np.nan

    if int(round(flag)) == 0 and not np.isnan(delta) and delta != 0:
        normalized_threshold = (threshold - value) / delta
        if normalized_threshold <= -N_SIGMA:
            split_proba = 0
        elif normalized_threshold >= N_SIGMA:
            split_proba = 1
        else:
            x = np.searchsorted(
                a=X_GAUS,
                v=normalized_threshold,
            )
            split_proba = GAUS[x]
    elif int(round(flag)) != 0 and not np.isnan(delta) and delta != 0:
        normalized_threshold = (threshold - flag) / delta
        if normalized_threshold <= 0:
            split_proba = 0
        elif normalized_threshold >= MAX_LM:
            split_proba = 1
        else:
            x = np.searchsorted(
                a=X_LM,
                v=normalized_threshold,
            )
            split_proba = LM_VALUES[x]
    else:
        if (threshold - value) >= 0:
            split_proba = 1
        elif (threshold - value) < 0:
            split_proba = 0

    return 1 - split_proba


@jit
def split_probability_all(values, deltas, flags, threshold):
    """
    Calculate split probabilities for all rows in values
    """

    nof_objcts = values.shape[0]
    ps = [
        split_probability(values[i], deltas[i], flags[i], threshold)
        for i in range(nof_objcts)
    ]
    ps = np.array(ps)

    return ps


@jit
def return_class_probas(pnode, pY):
    """
    The leaf probabilities for each class
    """

    nof_objects = pY.shape[0]
    nof_classes = pY.shape[1]
    class_probas = np.zeros(nof_classes)

    for i in range(nof_objects):
        class_probas += pnode[i] * pY[i, :]

    class_probas = class_probas / len(pnode)

    return class_probas


############################################################
############################################################
############################ MISC  #########################
############################################################
############################################################


@jit
def get_split_objects(
    pnode, p_split_right, p_split_left, is_max, n_objects_node, keep_proba
):
    pnode_right = pnode * p_split_right
    pnode_left = pnode * p_split_left

    pnode_right_tot = np.nansum(pnode_right)
    pnode_left_tot = np.nansum(pnode_left)
    pnode_tot = pnode_right_tot + pnode_left_tot

    is_nan = np.isnan(p_split_right)

    p_split_right_batch = pnode_right_tot / pnode_tot
    p_split_right[is_nan] = p_split_right_batch
    pnode_right[is_nan] = pnode[is_nan] * p_split_right[is_nan]

    p_split_left_batch = pnode_left_tot / pnode_tot
    p_split_left[is_nan] = p_split_left_batch
    pnode_left[is_nan] = pnode[is_nan] * p_split_left[is_nan]

    best_right = [0]
    best_left = [0]

    is_max_right = [0]
    is_max_left = [0]

    for i in range(n_objects_node):
        if p_split_right[i] >= 0.5 and is_max[i] == 1:
            best_right.append(i)
            is_max_right.append(1)
        elif pnode_right[i] > keep_proba:
            best_right.append(i)
            is_max_right.append(0)

        if p_split_left[i] > 0.5 and is_max[i] == 1:
            best_left.append(i)
            is_max_left.append(1)
        elif pnode_left[i] > keep_proba:
            best_left.append(i)
            is_max_left.append(0)

    best_right = np.array(best_right)
    best_left = np.array(best_left)
    is_max_right = np.array(is_max_right)
    is_max_left = np.array(is_max_left)

    pnode_right, _ = pull_values(pnode_right, best_right[1:], best_left[1:])
    _, pnode_left = pull_values(pnode_left, best_right[1:], best_left[1:])

    return (
        pnode_right,
        pnode_left,
        best_right[1:],
        best_left[1:],
        is_max_right[1:],
        is_max_left[1:],
        p_split_right_batch,
    )


# @jit
def choose_features(nof_features):
    """
    function randomly selects the features that will be examined for each split
    """
    features_indices = np.arange(nof_features)
    features_chosen = np.random.choice(
        features_indices, size=nof_features, replace=False
    )

    return features_chosen


# @jit
def pull_values(A, right, left):
    """
    Splits an array A to two
    according to lists of indicies
    given in right and left
    """
    A_left = A[left]
    A_right = A[right]

    return A_right, A_left


def get_pY(pY_true, y_fake):
    """
    Recieves a vector with the probability to be true (pY_true)
    returns a matrix with the probability to be in each class

    we put pY_true as the probability of the true class
    and (1-pY_true)/(nof_lables-1) for all other classes
    """
    nof_objects = len(pY_true)

    all_labels = np.unique(y_fake)
    label_dict = {i: a for i, a in enumerate(all_labels)}
    nof_labels = len(all_labels)

    pY = np.zeros([nof_objects, nof_labels])

    for o in range(nof_objects):
        for c_idx, c in enumerate(all_labels):
            if y_fake[o] == c:
                pY[o, c_idx] = pY_true[o]
            else:
                pY[o, c_idx] = float(1 - pY_true[o]) / (nof_labels - 1)

    return pY, label_dict
