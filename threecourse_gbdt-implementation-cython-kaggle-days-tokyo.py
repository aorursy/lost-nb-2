#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, log_loss
from typing import List, Tuple, Optional

class Logger:
    
    def info(self, message: str):
        print(f"[{self.now_string()}]: {message}")

    def now_string(self: str):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

logger = Logger()    


# In[2]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[3]:


get_ipython().run_cell_magic('cython', '', 'import datetime\nimport logging\nimport numpy as np\nimport pandas as pd\nfrom sklearn.metrics import mean_squared_error, log_loss\nfrom typing import List, Tuple, Optional\n\ncimport cython\ncimport numpy as np\n\nctypedef np.npy_intp SIZE_t  # Type for indices and counters\nctypedef np.npy_float64 DOUBLE_t  # Type of y, sample_weight\n\n\nclass Data:\n\n    def __init__(self, x: np.ndarray, y: Optional[np.ndarray]):\n        self.values = x\n        self.target = y\n        self.sorted_indexes = None\n\n        # sort index for each feature\n        # note: necessary only for training\n        sorted_indexes = []\n        for feature_id in range(self.values.shape[1]):\n            sorted_indexes.append(np.argsort(self.values[:, feature_id]))\n        self.sorted_indexes = np.array(sorted_indexes).T\n\n\nclass Node:\n\n    def __init__(self, id: int, weight: float):\n        self.id = id\n\n        # note: necessary only for leaf node\n        self.weight = weight\n\n        # split information\n        self.feature_id = None\n        self.feature_value = None\n\n    def is_leaf(self) -> bool:\n        return self.feature_id is None\n\n\ncdef class TreeUtil:\n\n    @classmethod\n    def left_child_id(cls, id: int) -> int:\n        """node id of left child"""\n        return id * 2 + 1\n\n    @classmethod\n    def right_child_id(cls, id: int) -> int:\n        """node id of right child"""\n        return id * 2 + 2\n\n    @staticmethod\n    cdef DOUBLE_t loss(DOUBLE_t sum_grad, DOUBLE_t sum_hess):\n        # if np.isclose(sum_hess, 0.0, atol=1.e-8):\n        #     return None\n        return -0.5 * (sum_grad ** 2.0) / sum_hess\n\n    @staticmethod\n    cdef DOUBLE_t weight(DOUBLE_t sum_grad, DOUBLE_t sum_hess):\n        # if np.isclose(sum_hess, 0.0, atol=1.e-8):\n        #     return None\n        return -1.0 * sum_grad / sum_hess\n\n    @classmethod\n    def node_ids_depth(self, d: int) -> List[int]:\n        return list(range(2 ** d - 1, 2 ** (d + 1) - 1))\n\n\nclass Tree:\n\n    def __init__(self, params: dict):\n        self.params = params\n        self.nodes = []\n\n        # parameters\n        self.max_depth = params.get("max_depth")\n\n        # add initial node\n        node = Node(0, 0.0)\n        self.nodes.append(node)\n\n    def construct(self, data: Data, _grad: np.ndarray, _hess: np.ndarray):\n        cdef np.ndarray[DOUBLE_t, ndim=1] grad = _grad\n        cdef np.ndarray[DOUBLE_t, ndim=1] hess = _hess\n        cdef SIZE_t n\n        cdef np.ndarray[DOUBLE_t, ndim=2] values\n        cdef np.ndarray[SIZE_t, ndim=2] sorted_indexes\n        cdef np.ndarray[SIZE_t, ndim=1] sorted_index\n        cdef np.ndarray[SIZE_t, ndim=1] node_ids_data\n\n        cdef SIZE_t depth, node_id, feature_id, i, idx\n        cdef DOUBLE_t value, prev_value\n        cdef DOUBLE_t sum_grad, left_grad, right_grad\n        cdef DOUBLE_t sum_hess, left_hess, right_hess\n        cdef DOUBLE_t left_loss, right_loss\n        cdef DOUBLE_t gain\n        cdef SIZE_t best_feature_id\n        cdef DOUBLE_t best_gain, best_feature_value, best_left_weight, best_right_weight\n\n        grad = _grad\n        hess = _hess\n\n        # data\n        assert (data.sorted_indexes is not None)\n        n = len(data.values)\n        values = data.values.astype(\'double\')\n        sorted_indexes = data.sorted_indexes\n\n        # node ids records belong to\n        node_ids_data = np.zeros(n, dtype=int)\n\n        # for each depth\n        for depth in range(self.max_depth):\n\n            # node ids in the depth\n            node_ids_depth = TreeUtil.node_ids_depth(depth)\n\n            # 1. find best split ----------\n\n            # split information for each node\n            feature_ids, feature_values = [], []\n            left_weights, right_weights = [], []\n\n            # for each node\n            for node_id in node_ids_depth:\n\n                node = self.nodes[node_id]\n                # logger.debug(f"{node_id}: find split -----")\n\n                # sum grad and hess of the node\n                sum_grad, sum_hess = 0.0, 0.0\n                for i in range(n):\n                    if node_ids_data[i] != node_id:\n                        continue\n                    sum_grad += grad[i]\n                    sum_hess += hess[i]\n\n                # initial gain, which is all directed to left\n                best_gain, best_feature_id, best_feature_value = 0.0, 0, -np.inf\n                best_left_weight, best_right_weight = node.weight, 0.0\n\n                if sum_hess > 0:\n                    sum_loss = TreeUtil.loss(sum_grad, sum_hess)\n                else:\n                    sum_loss = 0.0\n\n                # logger.debug(f"sum grad:{sum_grad} hess:{sum_hess} loss:{sum_loss}")\n\n                # for each feature\n                for feature_id in range(data.values.shape[1]):\n                    prev_value = -np.inf\n                    left_grad, left_hess = 0.0, 0.0\n\n                    sorted_index = sorted_indexes[:, feature_id]\n\n                    # for each record\n                    for i in range(n):\n                        idx = sorted_index[i]\n                        # skip if the record does not belong to the node\n                        # NOTE: this calculation is redundant and inefficient.\n                        if node_ids_data[idx] != node_id:\n                            continue\n\n                        value = values[idx, feature_id]\n\n                        # evaluate split, if split can be made at the value\n                        if value != prev_value and left_hess > 0 and (sum_hess - left_hess) > 0:\n                            right_grad = sum_grad - left_grad\n                            right_hess = sum_hess - left_hess\n                            left_loss = TreeUtil.loss(left_grad, left_hess)\n                            right_loss = TreeUtil.loss(right_grad, right_hess)\n\n                            gain = sum_loss - (left_loss + right_loss)\n                            # logger.debug(f"\'feature{feature_id} < {value}\' " +\n                            #       f"lg:{left_grad:.3f} lh:{left_hess:.3f} rg:{right_grad:.3f} rh:{right_hess:.3f} " +\n                            #       f"ll:{left_loss:.3f} rl:{right_loss:.3f} gain:{gain:.3f}")\n                            if gain > best_gain:\n                                best_gain = gain\n                                best_feature_id = feature_id\n                                best_feature_value = value\n                                best_left_weight = TreeUtil.weight(left_grad, left_hess)\n                                best_right_weight = TreeUtil.weight(right_grad, right_hess)\n\n                        prev_value = value\n                        left_grad += grad[idx]\n                        left_hess += hess[idx]\n\n                # logger.debug(f"node_id:{node_id} split - \'feature{best_feature_id} < {best_feature_value}\'")\n                feature_ids.append(best_feature_id)\n                feature_values.append(best_feature_value)\n                left_weights.append(best_left_weight)\n                right_weights.append(best_right_weight)\n\n            # 2. update nodes and create new nodes ----------\n            for i in range(len(node_ids_depth)):\n                node_id = node_ids_depth[i]\n                feature_id = feature_ids[i]\n                feature_value = feature_values[i]\n                left_weight = left_weights[i]\n                right_weight = right_weights[i]\n\n                # update current node\n                node = self.nodes[node_id]\n                node.feature_id = feature_id\n                node.feature_value = feature_value\n\n                # create new nodes\n                left_node = Node(TreeUtil.left_child_id(node_id), left_weight)\n                right_node = Node(TreeUtil.right_child_id(node_id), right_weight)\n                self.nodes += [left_node, right_node]\n\n            # 3. update node ids of records----------\n            for i in range(len(node_ids_data)):\n                # directed by split\n                node_id = node_ids_data[i]\n                node = self.nodes[node_id]\n                feature_id, feature_value = node.feature_id, node.feature_value\n\n                # update\n                is_left = values[i, feature_id] < feature_value\n                if is_left:\n                    next_node_id = TreeUtil.left_child_id(node_id)\n                else:\n                    next_node_id = TreeUtil.right_child_id(node_id)\n                node_ids_data[i] = next_node_id\n\n    def predict(self, x: np.ndarray) -> np.ndarray:\n        cdef SIZE_t depth, i\n        cdef SIZE_t node_id, next_node_id\n        cdef DOUBLE_t feature_value\n        cdef np.ndarray[DOUBLE_t, ndim=2] values\n        cdef np.ndarray[SIZE_t, ndim=1] node_ids_data\n\n        values = x.astype("double")\n\n        # node ids records belong to\n        node_ids_data = np.zeros(len(values), dtype=int)\n\n        for depth in range(self.max_depth):\n            for i in range(len(node_ids_data)):\n                # directed by split\n                node_id = node_ids_data[i]\n                node = self.nodes[node_id]\n                feature_id, feature_value = node.feature_id, node.feature_value\n\n                # update\n                if feature_id is None:\n                    next_node_id = node_id\n                elif values[i, feature_id] < feature_value:\n                    next_node_id = TreeUtil.left_child_id(node_id)\n                else:\n                    next_node_id = TreeUtil.right_child_id(node_id)\n                node_ids_data[i] = next_node_id\n\n        weights = np.array([self.nodes[node_id].weight for node_id in node_ids_data])\n\n        return weights\n\n    def dump(self) -> str:\n        """dump tree information"""\n        ret = []\n        for depth in range(self.max_depth + 1):\n            node_ids_depth = TreeUtil.node_ids_depth(depth)\n            for node_id in node_ids_depth:\n                node = self.nodes[node_id]\n                if node.is_leaf():\n                    ret.append(f"{node_id}:leaf={node.weight}")\n                else:\n                    ret.append(\n                        f"{node_id}:[f{node.feature_id}<{node.feature_value}] " +\n                        f"yes={TreeUtil.left_child_id(node_id)},no={TreeUtil.right_child_id(node_id)}")\n        return "\\n".join(ret)')


# In[4]:


class GBDTEstimator:

    def __init__(self, params: dict):
        self.params: dict = params
        self.trees: List[Tree] = []

        # parameters
        self.n_round: int = params.get("n_round")
        self.eta: float = params.get("eta")

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray)             -> Tuple[np.ndarray, np.ndarray]:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        data = Data(x, y)
        self._fit(data)

    def _fit(self, data: Data):
        pred = np.zeros(len(data.values))
        for round in range(self.n_round):
            logger.info(f"construct tree[{round}] --------------------")
            grad, hess = self.calc_grad(data.target, pred)
            tree = Tree(self.params)
            tree.construct(data, grad, hess)
            self.trees.append(tree)
            # NOTE: predict only last tree
            pred += self._predict_last_tree(data)

    def predict(self, x: np.ndarray) -> np.ndarray:
        data = Data(x, None)
        return self._predict(data)

    def _predict(self, data: Data) -> np.ndarray:
        pred = np.zeros(len(data.values))
        for tree in self.trees:
            pred += tree.predict(data.values) * self.eta
        return pred

    def _predict_last_tree(self, data: Data) -> np.ndarray:
        assert(len(self.trees) > 0)
        tree = self.trees[-1]
        return tree.predict(data.values) * self.eta

    def dump_model(self) -> str:
        ret = []
        for i, tree in enumerate(self.trees):
            ret.append(f"booster[{i}]")
            ret.append(tree.dump())
        return "\n".join(ret)


class GBDTRegressor(GBDTEstimator):

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray)             -> Tuple[np.ndarray, np.ndarray]:
        grad = y_pred - y_true
        hess = np.ones(len(y_true))
        return grad, hess


class GBDTClassifier(GBDTEstimator):

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray)             -> Tuple[np.ndarray, np.ndarray]:
        # (reference) regression_loss.h
        y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
        eps = 1e-16
        grad = y_pred_prob - y_true
        hess = np.maximum(y_pred_prob * (1.0 - y_pred_prob), eps)
        return grad, hess

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # apply sigmoid
        return 1.0 / (1.0 + np.exp(-self.predict(x)))


# In[5]:


# setting parameters for experiment
data_rows = 1000
parameter_type = "base"


# In[6]:


data = pd.read_csv("../input/otto-group-product-classification-challenge/train.csv")
data["target"] = data["target"].str[-1:].astype(int)
# print(data["target"].value_counts())
data["target"] = np.where(data["target"] > 5, 1, 0)
data = data.drop("id", axis=1)

rand = np.random.RandomState(seed=71)
idx_all = rand.choice(len(data), data_rows * 2, replace=False)
idx = rand.choice(data_rows * 2, data_rows, replace=False)
mask_tr = np.isin(np.arange(data_rows * 2), idx)
mask_va = ~mask_tr
idx_tr = idx_all[mask_tr]
idx_va = idx_all[mask_va]

data_tr = data.iloc[idx_tr]
data_va = data.iloc[idx_va]

assert(data_tr.shape[0] == data_rows)
assert(data_va.shape[0] == data_rows)
assert(len(np.unique(np.concatenate([data_tr.index, data_va.index]))) == 2 * data_rows)

data_tr = data_tr.reset_index(drop=True)
data_va = data_va.reset_index(drop=True)


# In[7]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import log_loss\nimport time\n\ntr_x = data_tr.drop("target", axis=1).values\nva_x = data_va.drop("target", axis=1).values\ntr_y = data_tr["target"].values\nva_y = data_va["target"].values\n\nif parameter_type == "simple":\n    params = {"n_round": 2, "max_depth": 2, "eta": 1.0}\nif parameter_type == "base":\n    params = {"n_round": 25, "max_depth": 5, "eta": 0.1}\n\nstart_time = time.time()\nmodel = GBDTClassifier(params)\nmodel.fit(tr_x, tr_y)\nend_time = time.time()\nlogger.info(f"elapsed_time: {end_time - start_time:.2f} sec")\nva_pred = model.predict_proba(va_x)\nscore = log_loss(va_y, va_pred)\nlogger.info(f"logloss: {score}")')


# In[8]:


get_ipython().run_cell_magic('time', '', 'import xgboost as xgb\n\nsimple_params = {"n_round": 2, "max_depth": 2,\n                 "objective": "binary:logistic",\n                 "base_score": 0.5, "random_state": 71, "seed": 171,\n                 "eta": 1.0, "alpha": 0.0, "lambda": 0.0, "tree_method": "exact",\n                 "colsample_bytree": 1.0, "subsample": 1.0,\n                 "gamma": 0.0, "min_child_weight": 0.0, "nthread": 1, "early_stopping_rounds":100}\n\nbase_params = {\n    "objective": "binary:logistic",\n    "eta": 0.1,\n    "gamma": 0.0, "alpha": 0.0, "lambda": 1.0,\n    "min_child_weight": 1, "max_depth": 5,\n    "subsample": 0.8, "colsample_bytree": 0.8,\n    \'silent\': 1, \'random_state\': 71,\n    "n_round": 1000, "early_stopping_rounds": 10,\n    "nthread": 1,\n}\n\n\nif parameter_type == "simple":\n    params = simple_params\nif parameter_type == "base":\n    params = base_params\n\ntr_x = data_tr.drop("target", axis=1).values\nva_x = data_va.drop("target", axis=1).values\ntr_y = data_tr["target"].values\nva_y = data_va["target"].values\ndtrain = xgb.DMatrix(tr_x, tr_y)\ndvalid = xgb.DMatrix(va_x, va_y)\n\nn_round = params.pop("n_round")\nearly_stopping_rounds = params.pop("early_stopping_rounds")\n\nwatchlist = [(dtrain, \'train\'), (dvalid, \'eval\')]\n\nbst = xgb.train(params, dtrain, num_boost_round=n_round,\n                evals=watchlist, early_stopping_rounds=early_stopping_rounds)\nva_pred = bst.predict(dvalid)\nscore = log_loss(va_y, va_pred)\nprint(score)\n#bst.dump_model("model_otto.txt")')

