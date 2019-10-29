import math
import numpy as np
class KNeighborsClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        if (X_train is None or y_train is None or len(X_train) != len(y_train)):
            raise Exception("incorrect input")
        self.X = X_train
        self.y = y_train

    def predict(self, X_test):
        if (X_test is None or len(X_test[0]) != len(self.X[0])):
            raise Exception("incorrect input")
        dArr = [float('inf')] * self.k
        index = [-1] * self.k
        idxNum = len(set(self.y))
        y_pred = [-1] * len(X_test)
        for j in range(len(X_test)):
            for i in range(len(self.X)):
                dis = self.distance(self.X[i], X_test[j])
                if dis < max(dArr):
                    replacei = dArr.index(max(dArr))
                    dArr[replacei] = dis
                    index[replacei] = self.y[i]
            for idx in range(idxNum):
                if index.count(idx) / self.k >= 1 / idxNum and index.count(idx) > y_pred[j] :
                    y_pred[j] = idx
        return y_pred


    def score(self, y_pred_test, y_test):
        if (y_pred_test is None or y_test is None or len(y_pred_test) != len(y_test)):
            raise Exception("incorrect input")
        match = np.sum(np.array(y_pred_test) == np.array(y_test))
        return match / len(y_pred_test)

    def distance(self, a, b):
        if a is None or b is None or len(a) != len(b):
            raise Exception("Two points should in the same dimension!")
        return np.linalg.norm(a - b)
