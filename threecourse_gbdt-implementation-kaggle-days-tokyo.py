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


class Data:

    def __init__(self, x: np.ndarray, y: Optional[np.ndarray]):
        self.values: np.array = x
        self.target: Optional[np.array] = y
        self.sorted_indexes: Optional[np.array] = None

        # sort index for each feature
        # note: necessary only for training
        sorted_indexes = []
        for feature_id in range(self.values.shape[1]):
            sorted_indexes.append(np.argsort(self.values[:, feature_id]))
        self.sorted_indexes = np.array(sorted_indexes).T


class Node:

    def __init__(self, id: int, weight: float):
        self.id: int = id

        # note: necessary only for leaf node
        self.weight: float = weight

        # split information
        self.feature_id: int = None
        self.feature_value: float = None

    def is_leaf(self) -> bool:
        return self.feature_id is None


class TreeUtil:

    @classmethod
    def left_child_id(cls, id: int) -> int:
        """node id of left child"""
        return id * 2 + 1

    @classmethod
    def right_child_id(cls, id: int) -> int:
        """node id of right child"""
        return id * 2 + 2

    @classmethod
    def loss(cls, sum_grad: float, sum_hess: float) -> Optional[float]:
        if np.isclose(sum_hess, 0.0, atol=1.e-8):
            return None
        return -0.5 * (sum_grad ** 2.0) / sum_hess

    @classmethod
    def weight(cls, sum_grad: float, sum_hess: float) -> Optional[float]:
        if np.isclose(sum_hess, 0.0, atol=1.e-8):
            return None
        return -1.0 * sum_grad / sum_hess

    @classmethod
    def node_ids_depth(self, d: int) -> List[int]:
        return list(range(2 ** d - 1, 2 ** (d + 1) - 1))


class Tree:

    def __init__(self, params: dict):
        self.params: dict = params
        self.nodes: List[Node] = []

        # parameters
        self.max_depth: int = params.get("max_depth")

        # add initial node
        node = Node(0, 0.0)
        self.nodes.append(node)

    def construct(self, data: Data, grad: np.ndarray, hess: np.ndarray):

        # data
        assert (data.sorted_indexes is not None)
        n = len(data.values)
        values = data.values
        sorted_indexes = data.sorted_indexes

        # node ids records belong to
        node_ids_data = np.zeros(n, dtype=int)

        # [comment with [] is important for understanding]
        # [for each depth]
        for depth in range(self.max_depth):

            # node ids in the depth
            node_ids_depth = TreeUtil.node_ids_depth(depth)

            # [1. find best split] -------------------

            # split information for each node
            feature_ids, feature_values = [], []
            left_weights, right_weights = [], []

            # [for each node]
            for node_id in node_ids_depth:

                node = self.nodes[node_id]
                # logger.debug(f"{node_id}: find split -----")

                # [calculate sum grad and hess of records in the node]
                sum_grad, sum_hess = 0.0, 0.0
                for i in range(n):
                    if node_ids_data[i] != node_id:
                        continue
                    sum_grad += grad[i]
                    sum_hess += hess[i]

                # [initialize best gain (set as all directed to left)]
                best_gain, best_feature_id, best_feature_value = 0.0, 0, -np.inf
                best_left_weight, best_right_weight = node.weight, 0.0

                if sum_hess > 0:
                    sum_loss = TreeUtil.loss(sum_grad, sum_hess)
                else:
                    sum_loss = 0.0

                # logger.debug(f"sum grad:{sum_grad} hess:{sum_hess} loss:{sum_loss}")

                # [for each feature]
                for feature_id in range(data.values.shape[1]):
                    prev_value = -np.inf
                    
                    # [have gradients/hessian of left child records(value of record < value of split)]
                    left_grad, left_hess = 0.0, 0.0

                    sorted_index = sorted_indexes[:, feature_id]

                    # [for each sorted record]
                    for i in sorted_index:
                        # skip if the record does not belong to the node
                        # NOTE: this calculation is redundant and inefficient.
                        if node_ids_data[i] != node_id:
                            continue

                        value = values[i, feature_id]

                        # [evaluate split, if split can be made at the record]
                        if value != prev_value and left_hess > 0 and (sum_hess - left_hess) > 0:
                            
                            # [calculate loss of the split using gradient and hessian]
                            right_grad = sum_grad - left_grad
                            right_hess = sum_hess - left_hess
                            left_loss = TreeUtil.loss(left_grad, left_hess)
                            right_loss = TreeUtil.loss(right_grad, right_hess)
                            if left_loss is not None and right_loss is not None:
                                gain = sum_loss - (left_loss + right_loss)
                                # logger.debug(f"'feature{feature_id} < {value}' " +
                                #       f"lg:{left_grad:.3f} lh:{left_hess:.3f} rg:{right_grad:.3f} rh:{right_hess:.3f} " +
                                #       f"ll:{left_loss:.3f} rl:{right_loss:.3f} gain:{gain:.3f}")
                                
                                # [update if the gain is better than current best gain]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_feature_id = feature_id
                                    best_feature_value = value
                                    best_left_weight = TreeUtil.weight(left_grad, left_hess)
                                    best_right_weight = TreeUtil.weight(right_grad, right_hess)

                        prev_value = value
                        left_grad += grad[i]
                        left_hess += hess[i]

                # logger.debug(f"node_id:{node_id} split - 'feature{best_feature_id} < {best_feature_value}'")
                feature_ids.append(best_feature_id)
                feature_values.append(best_feature_value)
                left_weights.append(best_left_weight)
                right_weights.append(best_right_weight)

            # [2. update nodes and create new nodes] ----------
            for i in range(len(node_ids_depth)):
                node_id = node_ids_depth[i]
                feature_id = feature_ids[i]
                feature_value = feature_values[i]
                left_weight = left_weights[i]
                right_weight = right_weights[i]

                # update current node
                node = self.nodes[node_id]
                node.feature_id = feature_id
                node.feature_value = feature_value

                # create new nodes
                left_node = Node(TreeUtil.left_child_id(node_id), left_weight)
                right_node = Node(TreeUtil.right_child_id(node_id), right_weight)
                self.nodes += [left_node, right_node]

            # [3. update node ids of records] ----------
            for i in range(len(node_ids_data)):
                # directed by split
                node_id = node_ids_data[i]
                node = self.nodes[node_id]
                feature_id, feature_value = node.feature_id, node.feature_value

                # update
                is_left = values[i, feature_id] < feature_value
                if is_left:
                    next_node_id = TreeUtil.left_child_id(node_id)
                else:
                    next_node_id = TreeUtil.right_child_id(node_id)
                node_ids_data[i] = next_node_id

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = x

        # node ids records belong to
        node_ids_data = np.zeros(len(values), dtype=int)

        for depth in range(self.max_depth):
            for i in range(len(node_ids_data)):
                # directed by split
                node_id = node_ids_data[i]
                node = self.nodes[node_id]
                feature_id, feature_value = node.feature_id, node.feature_value

                # update
                if feature_id is None:
                    next_node_id = node_id
                elif values[i, feature_id] < feature_value:
                    next_node_id = TreeUtil.left_child_id(node_id)
                else:
                    next_node_id = TreeUtil.right_child_id(node_id)
                node_ids_data[i] = next_node_id

        weights = np.array([self.nodes[node_id].weight for node_id in node_ids_data])

        return weights

    def dump(self) -> str:
        """dump tree information"""
        ret = []
        for depth in range(self.max_depth + 1):
            node_ids_depth = TreeUtil.node_ids_depth(depth)
            for node_id in node_ids_depth:
                node = self.nodes[node_id]
                if node.is_leaf():
                    ret.append(f"{node_id}:leaf={node.weight}")
                else:
                    ret.append(
                        f"{node_id}:[f{node.feature_id}<{node.feature_value}] " +
                        f"yes={TreeUtil.left_child_id(node_id)},no={TreeUtil.right_child_id(node_id)}")
        return "\n".join(ret)


# In[3]:


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


# In[4]:


# setting parameters for experiment
data_rows = 1000
parameter_type = "base"


# In[5]:


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


# In[6]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import log_loss\nimport time\n\ntr_x = data_tr.drop("target", axis=1).values\nva_x = data_va.drop("target", axis=1).values\ntr_y = data_tr["target"].values\nva_y = data_va["target"].values\n\nif parameter_type == "simple":\n    params = {"n_round": 2, "max_depth": 2, "eta": 1.0}\nif parameter_type == "base":\n    params = {"n_round": 25, "max_depth": 5, "eta": 0.1}\n\nstart_time = time.time()\nmodel = GBDTClassifier(params)\nmodel.fit(tr_x, tr_y)\nend_time = time.time()\nlogger.info(f"elapsed_time: {end_time - start_time:.2f} sec")\nva_pred = model.predict_proba(va_x)\nscore = log_loss(va_y, va_pred)\nlogger.info(f"logloss: {score}")')


# In[7]:


get_ipython().run_cell_magic('time', '', 'import xgboost as xgb\n\nsimple_params = {"n_round": 2, "max_depth": 2,\n                 "objective": "binary:logistic",\n                 "base_score": 0.5, "random_state": 71, "seed": 171,\n                 "eta": 1.0, "alpha": 0.0, "lambda": 0.0, "tree_method": "exact",\n                 "colsample_bytree": 1.0, "subsample": 1.0,\n                 "gamma": 0.0, "min_child_weight": 0.0, "nthread": 1, "early_stopping_rounds":100}\n\nbase_params = {\n    "objective": "binary:logistic",\n    "eta": 0.1,\n    "gamma": 0.0, "alpha": 0.0, "lambda": 1.0,\n    "min_child_weight": 1, "max_depth": 5,\n    "subsample": 0.8, "colsample_bytree": 0.8,\n    \'silent\': 1, \'random_state\': 71,\n    "n_round": 1000, "early_stopping_rounds": 10,\n    "nthread": 1,\n}\n\n\nif parameter_type == "simple":\n    params = simple_params\nif parameter_type == "base":\n    params = base_params\n\ntr_x = data_tr.drop("target", axis=1).values\nva_x = data_va.drop("target", axis=1).values\ntr_y = data_tr["target"].values\nva_y = data_va["target"].values\ndtrain = xgb.DMatrix(tr_x, tr_y)\ndvalid = xgb.DMatrix(va_x, va_y)\n\nn_round = params.pop("n_round")\nearly_stopping_rounds = params.pop("early_stopping_rounds")\n\nwatchlist = [(dtrain, \'train\'), (dvalid, \'eval\')]\n\nbst = xgb.train(params, dtrain, num_boost_round=n_round,\n                evals=watchlist, early_stopping_rounds=early_stopping_rounds)\nva_pred = bst.predict(dvalid)\nscore = log_loss(va_y, va_pred)\nprint(score)\n#bst.dump_model("model_otto.txt")\n\n# n = 1000: 25 rounds, logloss:0.2439\n# n = 30000: 150 rounds, 13sec, logloss:0.1446')

