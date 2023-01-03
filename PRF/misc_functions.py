import numpy
from numba import jit
from scipy.stats import norm

############################################################
############################################################
# Modified script from here
# Importo l'integratore per la distribuzione cumulativa
from scipy.integrate import quad

K = 10

# Precalcola la distribuzione. Gli estremi vanno sempre da zero
#  fino al parametro calcolato sui fogli (vedo cosetto di cartoncino)
MAX_LM = K
X_LM = numpy.arange(0, MAX_LM, 0.01)


# Costruisco la distribuzione cumulativa
def LMDist(x):
    return 1 / (K + numpy.e**(-K) - 1) * (1 - numpy.e**(-x))


def LMCumDist(u):
    return quad(LMDist, 0, u)[0]


# Trasformo nella versione vettorizzata
vectLMCumDist = numpy.vectorize(LMCumDist)

# Calcolo sui valori tabulati
LM_VALUES = vectLMCumDist(X_LM)
LM_VALUES = numpy.append(LM_VALUES, 1)

# Introduco le liste per indentificare cosa è magnitudine limite. Per info vedi il file corrispondente
# In questa versione le magnitudini limite NON servono: per l'ordine vedi il vecchio script

print("K Value: {}".format(K))
print("Using the new module")

# Original script starts here
cache = True

############################################################
############################################################
############ Propogate Probabilities functions  ############
############################################################
############################################################

# Parametri gaussiana, per tutto il resto. Infittisco un poco la
#  griglia, in questo momento mi pare troppo lassa
N_SIGMA = 3
X_GAUS = numpy.arange(-N_SIGMA, N_SIGMA, 0.01)
GAUS = numpy.array(norm(0, 1).cdf(X_GAUS))
GAUS = numpy.append(GAUS, 1)


@jit(cache=cache, nopython=True)
def split_probability(value, delta, flag, threshold):
    """
    Calculate split probability for a single object
    """
    # flag != 0 -> upper limit (carries info for turnover)

    if numpy.isnan(value):
        return numpy.nan

    if int(round(flag)) == 0 and not numpy.isnan(delta) and delta != 0:
        normalized_threshold = (threshold - value) / delta
        if normalized_threshold <= -N_SIGMA:
            split_proba = 0
        elif normalized_threshold >= N_SIGMA:
            split_proba = 1
        else:
            x = numpy.searchsorted(
                a=X_GAUS,
                v=normalized_threshold,
            )
            split_proba = GAUS[x]
    elif int(round(flag)) != 0 and not numpy.isnan(delta) and delta != 0:
        normalized_threshold = (threshold - flag) / delta
        if normalized_threshold <= 0:
            split_proba = 0
        elif normalized_threshold >= MAX_LM:
            split_proba = 1
        else:
            x = numpy.searchsorted(
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


@jit(cache=cache, nopython=True)
def split_probability_all(values, deltas, flags, threshold):
    """
    Calculate split probabilities for all rows in values
    """

    nof_objcts = values.shape[0]
    ps = [
        split_probability(values[i], deltas[i], flags[i], threshold)
        for i in range(nof_objcts)
    ]
    ps = numpy.array(ps)

    return ps


@jit(cache=cache, nopython=True)
def return_class_probas(pnode, pY):
    """
    The leaf probabilities for each class
    """

    nof_objects = pY.shape[0]
    nof_classes = pY.shape[1]
    class_probas = numpy.zeros(nof_classes)

    for i in range(nof_objects):
        class_probas += pnode[i] * pY[i, :]

    # class_probas = class_probas/numpy.sum(pnode)
    class_probas = class_probas / len(pnode)
    # class_probas = pY

    return class_probas


############################################################
############################################################
############################ MISC  #########################
############################################################
############################################################


@jit(cache=True, nopython=True)
def get_split_objects(pnode, p_split_right, p_split_left, is_max,
                      n_objects_node, keep_proba):

    pnode_right = pnode * p_split_right
    pnode_left = pnode * p_split_left

    pnode_right_tot = numpy.nansum(pnode_right)
    pnode_left_tot = numpy.nansum(pnode_left)
    pnode_tot = pnode_right_tot + pnode_left_tot

    is_nan = numpy.isnan(p_split_right)

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
        # if is_nan[i]:
        #    best_right.append(i)
        #    best_left.append(i)
        #    if (is_max[i] == 1):
        #        if (p_split_right_batch > p_split_left_batch):
        #            is_max_right.append(1)
        #            is_max_left.append(0)
        #        else:
        #            is_max_right.append(0)
        #            is_max_left.append(1)
        # else:
        if (p_split_right[i] >= 0.5 and is_max[i] == 1):
            best_right.append(i)
            is_max_right.append(1)
        elif pnode_right[i] > keep_proba:
            best_right.append(i)
            is_max_right.append(0)

        if (p_split_left[i] > 0.5 and is_max[i] == 1):
            best_left.append(i)
            is_max_left.append(1)
        elif pnode_left[i] > keep_proba:
            best_left.append(i)
            is_max_left.append(0)

    best_right = numpy.array(best_right)
    best_left = numpy.array(best_left)
    is_max_right = numpy.array(is_max_right)
    is_max_left = numpy.array(is_max_left)

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


#@jit(cache=True, nopython=True)
def choose_features(nof_features, max_features):
    """
    function randomly selects the features that will be examined for each split
    """
    features_indices = numpy.arange(nof_features)
    #numpy.random.seed()
    #features_chosen = numpy.random.choice(features_indices, size=max_features, replace = True)
    features_chosen = numpy.random.choice(features_indices,
                                          size=nof_features,
                                          replace=False)

    #print(features_chosen)
    return features_chosen


@jit(cache=True, nopython=True)
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

    all_labels = numpy.unique(y_fake)
    label_dict = {i: a for i, a in enumerate(all_labels)}
    nof_labels = len(all_labels)

    pY = numpy.zeros([nof_objects, nof_labels])

    for o in range(nof_objects):
        for c_idx, c in enumerate(all_labels):
            if y_fake[o] == c:
                pY[o, c_idx] = pY_true[o]
            else:
                pY[o, c_idx] = float(1 - pY_true[o]) / (nof_labels - 1)

    return pY, label_dict
