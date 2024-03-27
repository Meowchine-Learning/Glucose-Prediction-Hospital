import skops.io as sio
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
# import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


class GlucosePredictionModel:
    def __init__(self):
        """ Dashboard """
        # Data Preparations
        self.X = None
        self.y = None
        self.dataInput = "../../features/output/FormalizedDATA.npy"

        # Model Settings
        self.model = ["rt", "rf", "gbdt"][0]
        self.optimalModel = None
        self.toReload = False
        self.path = "/Users/den/Desktop/CMPUT 469 - Proj/repo/Glucose-Prediction-Hospital/src/models/tree&forests/"

        # Learning Settings
        self.trainingWindowSize = 100
        self.validationWindowSize = 20
        self.rollingStep = 80
        self.plotRange = 10000

    def run(self):
        self.modelInput(self.dataInput)
        if self.toReload:
            self.modelReload(self.path + f"pretrained/{self.model}.pkl")
        else:
            self.modelTraining(getReport=True, saveModel=True)
        self.modelTest(getReport=True)
        # self.modelOutput(outputPath="../../features/output/PredictedDATA.npy", predictions)

    def modelInput(self, inputPath):
        data = np.load(inputPath, allow_pickle=True)
        if data.shape[1] < 2:
            raise ValueError()

        _ = data[:, 0]
        self.X = np.array(data[:, 10:], dtype=np.float32)
        self.y = np.array(data[:, 1:10], dtype=np.float32)

    def modelOutput(self, outputPath, DATA):
        np.save(outputPath, np.array(DATA))
        print(f"\t> Formalized Data printed to {outputPath}...")

    def modelTraining(self, getReport=True, saveModel=True):
        y_885 = self.y[:, 4]
        y_885 = y_885.reshape(-1)

        y_true_values = []
        y_predicted_values = []
        modelRecorder = []
        errorRecorder = []

        # Rolling-origin Validation:
        for TRAIN_START in tqdm(range(0, len(self.X) - self.trainingWindowSize, self.rollingStep)):
            TRAIN_END = TRAIN_START + self.trainingWindowSize
            VALIDATION_END = TRAIN_END + self.validationWindowSize

            # Data for Training & Validation:
            X_train = self.X[TRAIN_START:TRAIN_END]
            y_train = y_885[TRAIN_START:TRAIN_END]
            X_eval = self.X[TRAIN_END:VALIDATION_END]
            y_eval = y_885[TRAIN_END:VALIDATION_END]

            # todo - Hyperparameters for fine-tuning
            MODELS = {"rt": DecisionTreeRegressor(criterion='absolute_error',
                                                  max_depth=[5, 10, 15][0],
                                                  min_samples_split=[2, 3, 4][0]),
                      "rf": RandomForestRegressor(criterion="entropy",
                                                  n_jobs=-1,
                                                  n_estimators=[100, 200, 300][0],
                                                  min_samples_split=[2, 3, 4][0]),
                      "gbdt": GradientBoostingRegressor(loss="absolute_error",
                                                        learning_rate=0.1,
                                                        n_estimators=[10, 20, 30][0],
                                                        max_depth=[3, 4, 5][0])}
            model = MODELS[self.model]

            model.fit(X_train, y_train)  # eval_metric='mae'
            y_predict = model.predict(X_eval)
            l1_error = mean_absolute_error(y_eval, y_predict)

            # print(f"In-window l1_error: {l1_error}")
            y_true_values.extend(y_eval)
            y_predicted_values.extend(y_predict)
            modelRecorder.append(model)
            errorRecorder.append(l1_error)

        averageError = np.mean(errorRecorder)
        optimalError = np.min(errorRecorder)
        worstError = np.max(errorRecorder)
        self.optimalModel = modelRecorder[errorRecorder.index(np.min(errorRecorder))]

        if saveModel:
            self.modelSave(modelPath := self.path + f'pretrained/{self.model}.pkl')
            print(f"\t\n>> Optimal model is stored at {modelPath}'")

        if getReport:
            instructions = f"[{str(self.trainingWindowSize)}+{str(self.validationWindowSize)}+{str(self.rollingStep)}]"
            validatePlotOutputPath = self.path + f'plots/{self.model} {instructions}({str(self.plotRange)})(VALIDATE,{averageError}).png'

            # Report for Training and Validation:
            print("\n\n Validation Report:")
            print(f"\t* Average l1_error over all windows: {averageError}")
            print(f"\t* Optimal l1_error over all windows: {optimalError}")
            print(f"\t* Worst l1_error over all windows: {worstError}")

            # Visualization for Training and Validation:
            plt.figure(figsize=(50, 6))
            x_axis = np.arange(len(y_true_values))
            plt.plot(x_axis[:self.plotRange], y_true_values[:self.plotRange], color='blue', label='True Values',
                     linewidth=1)
            plt.plot(x_axis[:self.plotRange], y_predicted_values[:self.plotRange], color='red', linestyle='--',
                     label='Predicted Values (VALIDATE)', linewidth=1)
            plt.title('True vs Predicted Values (VALIDATE)')
            plt.xlabel('Evaluation Point (hr, repeating exists)')
            plt.ylabel('Glucose Level (885)')
            plt.legend()
            plt.savefig(validatePlotOutputPath)
            plt.show()

    def modelTest(self, getReport=True):
        # Data for Test:
        y_885 = self.y[:, 4]
        y_885 = y_885.reshape(-1)

        # Model for Test:
        y_predict_all = self.optimalModel.predict(self.X)
        l1_error = mean_absolute_error(y_885, y_predict_all)

        # Report and Visualization for Test:
        if getReport:
            print("\n\n Test Report:")
            print(f"\t* l1_error in full data test: {l1_error}")
            plt.figure(figsize=(50, 6))
            x_axis = np.arange(len(y_885))
            plt.plot(x_axis[:self.plotRange], y_885[:self.plotRange], color='blue', label='True Values', linewidth=1)
            plt.plot(x_axis[:self.plotRange], y_predict_all[:self.plotRange], color='green', linestyle='--',
                     label='Predicted Values (TEST)', linewidth=1)
            plt.title('True vs Predicted Values (TEST)')
            plt.xlabel('Evaluation Point (hr)')
            plt.ylabel('Glucose Level (885)')
            plt.legend()
            instructions = f"[{str(self.trainingWindowSize)}+{str(self.validationWindowSize)}+{str(self.rollingStep)}]"
            plt.savefig(self.path + f'plots/{self.model} {instructions}({str(self.plotRange)})(TEST,{l1_error}).png')
            plt.show()

    def modelSave(self, filePath):
        sio.dump(self.optimalModel, filePath)

    def modelReload(self, filePath):
        self.optimalModel = sio.load(filePath, trusted=True)


if __name__ == '__main__':
    GlucosePredictionModel().run()
