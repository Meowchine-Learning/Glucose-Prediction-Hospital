import skops.io as sio
import numpy as np

from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


class GBDTRegressModel:
    def __init__(self, cv: int, cvEval: str):
        self.bestRegressor = None
        self.bestDepth = None
        self.bestN = None

        self.cvEval: str = cvEval
        self.cv: int = cv
        self.cvErrors: list = []
        self.bests: dict = {}

        # todo: Hyperparameters
        self.N = [10, 20, 30]
        self.depths = list(range(2, 6))
        self.minSampleLeaf = 5
        self.minSampleSplit = 10

    def fit(self, datasets_train, datasets_test, doTraining=True):
        if doTraining:
            self._training(datasets_train=datasets_train)
            self._evaluate()
            self._store()
        self._predict(datasets_test=datasets_test)

    def _training(self, datasets_train: tuple):
        X_train, y_train = datasets_train
        for n in self.N:
            errors = []
            for depth in self.depths:
                regressor = GradientBoostingRegressor(n_estimators=n,
                                                      learning_rate=0.1,
                                                      max_depth=depth,
                                                      min_samples_leaf=self.minSampleLeaf,
                                                      min_samples_split=self.minSampleSplit)
                regressor.fit(X_train, y_train)

                scores = cross_val_score(regressor, X_train, y_train,
                                         scoring=self.cvEval,
                                         cv=self.cv)
                errors.append(np.mean(-scores))
                self.bests[min(errors)] = [n, depth]
            self.cvErrors.append(errors)

        # Select the best parameters based on the lowest error
        self.bestN, self.bestDepth = self.bests[min(self.bests)]
        self.bestRegressor = GradientBoostingRegressor(n_estimators=self.bestN,
                                                       learning_rate=0.1,
                                                       max_depth=self.bestDepth,
                                                       min_samples_leaf=self.minSampleLeaf,
                                                       min_samples_split=self.minSampleSplit)
        self.bestRegressor.fit(X_train, y_train)

    def _store(self):
        sio.dump(self.bestRegressor, 'src/models/tree&forests/srcs/gbdt.pkl')

    def _reload(self):
        self.bestRegressor = sio.load('src/models/tree&forests/srcs/gbdt.pkl', trusted=True)

    def _predict(self, datasets_test: tuple):
        X_test, y_test = datasets_test
        y_pred = self.bestRegressor.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        print(f"> Test MAE: {test_mae:.4f}")

    def _evaluate(self):
        index = 0
        while index < len(self.cvErrors):
            n = self.N[index]

            plt.figure()
            plt.plot(self.depths, self.cvErrors[index], marker='o', linestyle='-')
            plt.xlabel('Max Depth')
            plt.ylabel('l1 Error')
            plt.title(f'NEstimators: {n}, CV MeanAbsoluteError')
            plt.savefig(f"eval/GBDTModel_n_{n}_cv_mae.png")
            index += 1


cvEval = 'neg_mean_absolute_error'
model = GBDTRegressModel(cv=5, cvEval=cvEval)

