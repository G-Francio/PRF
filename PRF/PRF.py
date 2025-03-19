import warnings

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from PRF import misc_functions as m
from PRF import tree

############################################################
############################################################
################ DecisionTreeClassifier Class  #############
############################################################
############################################################


class DecisionTreeClassifier:
    """
    This the the decision tree classifier class.
    """

    def __init__(
        self,
        criterion="gini",
        max_features=None,
        use_py_gini=True,
        use_py_leafs=True,
        max_depth=None,
        keep_proba=0.05,
        unsupervised=False,
        new_syn_data_frac=0,
        min_py_sum_leaf=1,
    ):
        self.criterion = criterion
        self.max_features = max_features
        self.use_py_gini = use_py_gini
        self.use_py_leafs = use_py_leafs
        self.max_depth = max_depth
        self.keep_proba = keep_proba
        self.is_node_arr_init = False
        self.unsupervised = unsupervised
        self.new_syn_data_frac = new_syn_data_frac
        self.min_py_sum_leaf = min_py_sum_leaf

    def get_nodes(self):
        node_list = []
        node_list = self.tree_.get_node_list(node_list, self.tree_, 0)[1]
        node_idx = np.zeros(len(node_list), dtype=int)
        for i, node in enumerate(node_list):
            node_idx[i] = node[0]
            node_list_sort = []
        new_order = np.argsort(node_idx)
        for idx in new_order:
            node_list_sort += [node_list[idx]]

        return node_list_sort

    def node_arr_init(self):
        if self.is_node_arr_init:
            return

        node_list = self.get_nodes()

        node_tree_results = np.ones([len(node_list), self.n_classes_]) * (-1)
        node_feature_idx = np.ones(len(node_list), dtype=int) * (-1)
        node_feature_th = np.zeros(len(node_list))
        node_true_branch = np.ones(len(node_list), dtype=int) * (-1)
        node_false_branch = np.ones(len(node_list), dtype=int) * (-1)
        node_p_right = np.zeros(len(node_list))

        for idx, n in enumerate(node_list):
            n = node_list[idx]
            if n[3] is not None:
                node_feature_idx[idx] = n[1]
                node_feature_th[idx] = n[2]
                node_true_branch[idx] = n[3]
                node_false_branch[idx] = n[4]
                node_p_right[idx] = n[6]
            else:
                node_tree_results[idx] = n[5]

        self.node_feature_idx = node_feature_idx
        self.node_feature_th = node_feature_th
        self.node_true_branch = node_true_branch
        self.node_false_branch = node_false_branch
        self.node_tree_results = node_tree_results
        self.node_p_right = node_p_right
        self.is_node_arr_init = True

        return

    def fit(self, X, pX, py, flags):
        """
        the DecisionTreeClassifier.fit() function with a similar appearance to that of sklearn
        """

        self.n_classes_ = py.shape[1]
        self.n_features_ = len(X[0])
        self.n_samples_ = len(X)
        self.feature_importances_ = [0] * self.n_features_
        self.is_node_arr_init = False

        pnode = np.ones(self.n_samples_)
        is_max = np.ones(self.n_samples_, dtype=int)

        py_flat = py.copy()
        py_flat[py < 0.5] = 0
        py_flat[py > 0.5] = 1

        if self.use_py_gini:
            py_gini = py
        else:
            py_gini = py_flat

        if self.use_py_leafs:
            py_leafs = py
        else:
            py_leafs = py_flat
        depth = 0

        self.tree_ = tree.fit_tree(
            X,
            pX,
            flags,
            py_gini,
            py_leafs,
            pnode,
            depth,
            is_max,
            self.max_depth,
            self.max_features,
            self.feature_importances_,
            self.n_samples_,
            self.keep_proba,
            self.unsupervised,
            self.new_syn_data_frac,
            self.min_py_sum_leaf,
        )

    def predict_proba(self, X, dX, flags, return_leafs=False):
        """
        The DecisionTreeClassifier.predict_proba() function with a similar appearance to the of sklearn
        """
        keep_proba = self.keep_proba

        result = tree.predict_all(
            self.node_tree_results,
            self.node_feature_idx,
            self.node_feature_th,
            self.node_true_branch,
            self.node_false_branch,
            self.node_p_right,
            X,
            dX,
            flags,
            keep_proba,
            return_leafs,
        )

        return result


############################################################
############################################################
################ RandomForestClassifier Class  #############
############################################################
############################################################


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=10,
        criterion="gini",
        max_features="auto",
        use_py_gini=True,
        use_py_leafs=True,
        max_depth=None,
        keep_proba=0.05,
        bootstrap=True,
        new_syn_data_frac=0,
        min_py_sum_leaf=1,
        n_jobs=1,
    ):
        self.n_estimators_ = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.estimators_ = []
        self.use_py_gini = use_py_gini
        self.use_py_leafs = use_py_leafs
        self.max_depth = max_depth
        self.keep_proba = keep_proba
        self.bootstrap = bootstrap
        self.new_syn_data_frac = new_syn_data_frac
        self.min_py_sum_leaf = min_py_sum_leaf
        self.n_jobs = n_jobs

    def check_input_X(self, X, dX, flags):
        if dX is None:
            dX = np.zeros(X.shape)

        if flags is None:
            flags = np.zeros(X.shape)

        dX[np.isnan(dX)] = 0
        X[np.isinf(dX)] = np.nan

        return X, dX, flags

    def _choose_objects(self, X, pX, py, flags):
        """
        function builds a sample of the same size as the input data, but chooses the objects with replacement
        according to the given probability matrix
        """
        nof_objects = py.shape[0]
        objects_indices = np.arange(nof_objects)
        objects_chosen = np.random.choice(objects_indices, nof_objects, replace=True)
        X_chosen = X[objects_chosen, :]
        pX_chosen = pX[objects_chosen, :]
        py_chosen = py[objects_chosen, :]
        flags_chosen = flags[objects_chosen, :]

        return X_chosen, pX_chosen, py_chosen, flags_chosen

    def _fit_single_tree(self, X, pX, py, flags):
        tree_ = DecisionTreeClassifier(
            criterion=self.criterion,
            max_features=self.max_features_num,
            use_py_gini=self.use_py_gini,
            use_py_leafs=self.use_py_leafs,
            max_depth=self.max_depth,
            keep_proba=self.keep_proba,
            unsupervised=self.unsupervised,
            new_syn_data_frac=self.new_syn_data_frac,
            min_py_sum_leaf=self.min_py_sum_leaf,
        )

        if self.bootstrap:
            X_chosen, pX_chosen, py_chosen, flags_chosen = self._choose_objects(
                X, pX, py, flags
            )
            tree_.fit(X_chosen, pX_chosen, py_chosen, flags_chosen)
        else:
            tree_.fit(X, pX, py, flags)

        return tree_

    def fit(self, X, dX=None, y=None, py=None, flags=None):
        """
        The RandomForestClassifier.fit() function with a similar appearance to that of sklearn
        """
        n_featuers = X.shape[1]
        n_objects = X.shape[0]
        self.n_features_ = n_featuers
        self.feature_importances_ = np.zeros(self.n_features_)

        if self.max_features == "auto" or self.max_features == "sqrt":
            self.max_features_num = int(np.sqrt(self.n_features_))
        elif self.max_features == "log2":
            self.max_features_num = int(np.log2(self.n_features_))
        elif type(self.max_features) is int:
            self.max_features_num = self.max_features
        elif type(self.max_features) is float:
            self.max_features_num = int(self.max_features * self.n_features_)
        else:
            self.max_features_num = self.n_features_

        # self.classes_ = list(np.sort(np.unique(y)))
        self.unsupervised = False
        if (py is None) and (y is None):
            self.unsupervised = True
            py = np.zeros([n_objects, 2])
            py[:, 0] = 1  # Class 'real'
            self.n_classes_ = 2
            self.label_dict = {i: i for i in range(self.n_classes_)}
        elif py is None:
            py, label_dict = m.get_pY(np.ones(len(y)), y)
            self.n_classes_ = py.shape[1]
            self.label_dict = label_dict
        elif y is None:
            self.n_classes_ = py.shape[1]
            self.label_dict = {i: i for i in range(self.n_classes_)}
        else:
            # Change this into a warning.warn, otherwise this piece of code
            #  is not even reachable
            warnings.warn("Both of {y, py} are given, ignoring y")
            self.n_classes_ = py.shape[1]
            self.label_dict = {i: i for i in range(self.n_classes_)}

        X, dX, flags = self.check_input_X(X, dX, flags)

        # Re-enable pooling
        if self.n_jobs == 1:
            tree_list = []
            for _ in tqdm(range(self.n_estimators_)):
                tree_list.append(self._fit_single_tree(X, dX, py, flags))
        else:
            tree_list = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._fit_single_tree)(X, dX, py, flags)
                for _ in range(self.n_estimators_)
            )

        self.estimators_ = []
        for tree_i in tree_list:
            self.estimators_.append(tree_i)
            self.feature_importances_ += np.array(tree_i.feature_importances_)

        self.feature_importances_ /= self.n_estimators_

        return self

    def predict_single_object(self, row):
        # let all the trees vote
        summed_probabilites = np.zeros(self.n_classes_)

        for tree_index in range(0, self.n_estimators_):
            y_proba_tree = self.estimators_[tree_index].predict_proba_one_object(row)
            summed_probabilites += y_proba_tree

        return np.argmax(summed_probabilites)

    def pick_best(self, class_proba):
        n_objects = class_proba.shape[0]
        new_class_proba = np.zeros(class_proba.shape)

        best_idx = np.argmax(class_proba, axis=1)
        for i in range(n_objects):
            new_class_proba[i, best_idx[i]] = class_proba[i, best_idx[i]]

        return new_class_proba

    def predict_single_tree(self, predict, X, dX, flags, out):
        # let all the trees vote - the function pick_best find the best class
        #  from each tree, and gives zero probability to all the others
        prediction = predict(X, dX, flags)

        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]

    def predict_proba(self, X, dX=None, flags=None):
        """
        The RandomForestClassifier.predict() function with a similar appearance to that of sklearn
        """
        proba = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        X, dX, flags = self.check_input_X(X, dX, flags)

        for _, tree_i in tqdm(enumerate(self.estimators_)):
            tree_i.node_arr_init()
            self.predict_single_tree(tree_i.predict_proba, X, dX, flags, proba)

        proba = [p / np.sum(p) for p in proba]

        return proba

    def apply(self, X, dX=None, flags=None):
        X, dX, flags = self.check_input_X(X, dX, flags)

        dX = np.zeros(X.shape)  # TODO: return leafs with probabilities for PRF

        for tree_i in self.estimators_:
            tree_i.node_arr_init()

        leafs = [
            tree_.predict_proba(X, dX, flags, return_leafs=True)[:, 0].reshape(-1, 1)
            for tree_ in self.estimators_
        ]
        leafs = np.hstack(leafs).astype(int)

        return leafs

    def predict(self, X, dX=None, flags=None):
        y_pred_inds = np.argmax(self.predict_proba(X, dX, flags), axis=1)
        y_pred = np.array([self.label_dict[i] for i in y_pred_inds])
        return y_pred

    def score(self, X, y, dX=None, flags=None):
        y_pred = self.predict(X, dX, flags)
        score = (y_pred == (y)).sum() / len(y)
        return score

    def __str__(self):
        sb = []
        do_not_print = [
            "estimators_",
            "use_py_gini",
            "use_py_leafs",
            "label_dict",
            "new_syn_data_frac",
        ]
        for key in self.__dict__:
            if key not in do_not_print:
                sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))

        sb = "ProbabilisticRandomForestClassifier(" + ", ".join(sb) + ")"
        return sb

    def __repr__(self):
        return self.__str__()
