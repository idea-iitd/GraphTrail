import numpy as np
import sympy
from pysr import PySRRegressor


class PySrModelBoolean:
    def __init__(self, train_ind_vec, test_ind_vec, train_pred, test_pred, weights, k=200):
        self.X_train = train_ind_vec
        self.X_test = test_ind_vec
        self.Y_train = train_pred
        self.Y_test = test_pred
        self.weights = weights
        self.kBest = k

    def fit(self):
        model3 = PySRRegressor(
            binary_operators=["Or(x, y) = ((x > zero(x)) | (y > zero(y))) * one(x)",
                              "And(x, y) = ((x > zero(x)) & (y > zero(y))) * one(x)",
                              "Xor(x, y) = (((x > 0) & (y <= 0)) | ((x <= 0) & (y > 0))) * 1f0"],
            unary_operators=["Not(x) = ~(x> zero(x)) * one(x)"],

            loss="loss(prediction, target) = sum(prediction != target)",

            extra_sympy_mappings={"Or": lambda x, y: sympy.Piecewise((1.0, (x > 0) | (y > 0)), (0.0, True)),
                                  "And": lambda x, y: sympy.Piecewise((1.0, (x > 0) & (y > 0)), (0.0, True)),
                                  "Not": lambda x: sympy.Piecewise((1.0, ~(x > 0)), (0.0, True)),
                                  "Xor": lambda x, y: sympy.Piecewise((1.0, (x > 0) ^ (y > 0)), (0.0, True))
                                  },

            select_k_features=min(self.kBest, 10),
            tempdir='./HallOfFame',
            temp_equation_file=True,
            delete_tempfiles=True,
            warm_start=False,
            batch_size=32,
            complexity_of_variables=0.1,
            weights=self.weights,
            complexity_of_operators={'Or': 0.1,
                                     'Not': 0.1, 'And': 0.1, 'Xor': 0.1},
            procs=32,
        )
        self.model = model3
        self.model.set_params(
            extra_sympy_mappings={"Or": lambda x, y: sympy.Piecewise((1.0, (x > 0) | (y > 0)), (0.0, True)),
                                  "And": lambda x, y: sympy.Piecewise((1.0, (x > 0) & (y > 0)), (0.0, True)),
                                  "Not": lambda x: sympy.Piecewise((1.0, ~(x > 0)), (0.0, True)),
                                  "Xor": lambda x, y: sympy.Piecewise((1.0, (x > 0) ^ (y > 0)), (0.0, True))
                                  }
        )
        self.model.fit(self.X_train, self.Y_train)
        self.model.refresh()
        return self.model

    def get_train_acc(self):
        self.model.refresh()
        Y_predicted = self.model.predict(self.X_train)
        train_acc = 0
        for i in range(len(self.Y_train)):
            if self.Y_train[i] == Y_predicted[i]:
                train_acc += 1

        train_acc /= len(self.Y_train)
        return train_acc

    def get_test_acc(self):
        self.model.refresh()
        Y_predicted = self.model.predict(self.X_test)
        test_acc = 0
        for i in range(len(self.Y_test)):
            if self.Y_test[i] == Y_predicted[i]:
                test_acc += 1

        test_acc /= len(self.Y_test)
        return test_acc


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


class PySrModelArithmetic:
    def __init__(self, train_ind_vec, test_ind_vec, train_pred, test_pred, weights):
        self.X_train = train_ind_vec
        self.X_test = test_ind_vec
        self.Y_train = train_pred
        self.Y_test = test_pred
        self.weights = weights

    def fit(self):
        model3 = PySRRegressor(
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["log", "exp", "abs", "sqrt", "relu", "round",
                             "sin", "cos", "tan", "sinh", "cosh", "tanh", "sign"],
            loss="loss(prediction, target) = abs(prediction - target)",
            select_k_features=10,
            tempdir='./HallOfFame',
            temp_equation_file=True,
            niterations=150,
            batch_size=32,
            weights=self.weights,
            complexity_of_variables=0.1,
            procs=4,
            turbo=True,
        )
        model3.fit(self.X_train, self.Y_train)
        self.model = model3
        return model3

    def get_train_loss(self):
        Y_predicted = self.model.predict(self.X_train)
        Y_prob = softmax(Y_predicted)
        loss = 0
        # binary cross entropy loss
        for i in range(len(self.Y_train)):
            pred_class = np.argmax(self.Y_train[i])
            loss += -np.log(Y_prob[i][pred_class])
        return loss

    def get_test_loss(self):
        Y_predicted = self.model.predict(self.X_test)
        Y_prob = softmax(Y_predicted)
        loss = 0
        # binary cross entropy loss
        for i in range(len(self.Y_test)):
            pred_class = np.argmax(self.Y_test[i])
            loss += -np.log(Y_prob[i][pred_class])
        return loss

    def get_train_acc(self):
        Y_predicted = self.model.predict(self.X_train)
        Y_prob = softmax(Y_predicted)
        train_acc = 0
        for i in range(len(self.Y_train)):
            pred_class = np.argmax(Y_prob[i])
            act_class = np.argmax(self.Y_train[i])
            if act_class == pred_class:
                train_acc += 1
        train_acc /= len(self.Y_train)
        return train_acc

    def get_test_acc(self):
        Y_predicted = self.model.predict(self.X_test)
        Y_prob = softmax(Y_predicted)
        test_acc = 0
        for i in range(len(self.Y_test)):
            pred_class = np.argmax(Y_prob[i])
            act_class = np.argmax(self.Y_test[i])
            if act_class == pred_class:
                test_acc += 1
        test_acc /= len(self.Y_test)
        return test_acc
