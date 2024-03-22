import skops.io as sio
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


FEATURES = [
    "UniqueSampleID",
    "LabTests",
    "#Day",
    "#Time",
    "Med",
    "Activity",
    "Nutrition",
    "Weight",
    "Height",
    "Age",
    "Sex",
    "Operations",
    "MedActs",
    "Diseases",
    "PriorMed"
]

# todo:
toReload = False        # todo
pretrainedModelOutputPath = 'pretrained/rfr.pkl'

training_window_size = 100
evaluation_window_size = 20
rolling_step = 1


def _store(trainedModel, filePath='pretrained/rt.pkl'):
    sio.dump(trainedModel, filePath)


def _reload(filePath='pretrained/rt.pkl'):
    return sio.load(filePath, trusted=True)


def _dataInput_npy(inputPath="../../features/output/FormalizedDATA.npy"):
    data = np.load(inputPath, allow_pickle=True)
    if data.shape[1] < 2:
        raise ValueError("The data must contain at least two columns for sampleID and y value.")

    uniqueSampleIDs = data[:, 0]
    y = np.array(data[:, 1:10], dtype=np.float32)
    X = np.array(data[:, 10:], dtype=np.float32)

    return uniqueSampleIDs, X, y


if __name__ == '__main__':
    _, X, y = _dataInput_npy()

    y = y[:, 4]
    y = y.reshape(-1)

    true_values = []
    predicted_values = []
    modelRecorder = []
    errorRecorder = []

    for TRAIN_START in tqdm(range(0, len(X) - training_window_size, rolling_step)):
        TRAIN_END = TRAIN_START + training_window_size
        TEST = TRAIN_END + evaluation_window_size

        X_train = X[TRAIN_START:TRAIN_END]
        y_train = y[TRAIN_START:TRAIN_END]
        X_eval = X[TRAIN_END:TEST]
        y_eval = y[TRAIN_END:TEST]

        #model = SVR()
        #model = DecisionTreeRegressor()
        #model = GradientBoostingRegressor()
        model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_predict = model.predict(X_eval)
        true_values.extend(y_eval)
        predicted_values.extend(y_predict)

        l1_error = mean_absolute_error(y_eval, y_predict)
        # print(f"In-window l1_error: {l1_error}")
        modelRecorder.append(model)
        errorRecorder.append(l1_error)

    averageError = np.mean(errorRecorder)
    optimalError = np.min(errorRecorder)
    worstError = np.max(errorRecorder)
    optimalModel = modelRecorder[errorRecorder.index(np.min(errorRecorder))]
    _store(optimalModel, pretrainedModelOutputPath)
    print(f"* Average l1_error over all windows: {averageError}")
    print(f"* Optimal l1_error over all windows: {optimalError}")
    print(f"* Worst l1_error over all windows: {worstError}")
    print(f"\n>> Optimal model is stored at {pretrainedModelOutputPath}'")

    plt.figure(figsize=(30, 6))
    x_axis = evaluation_points = np.arange(len(true_values))
    CUT = 5000
    # linewidth=1.5
    plt.plot(x_axis[:CUT], true_values[:CUT], color='blue', label='True Values')
    plt.plot(x_axis[:CUT], predicted_values[:CUT], color='red', linestyle='--', label='Predicted Values')
    plt.title('True vs Predicted Values')
    plt.xlabel('Evaluation Point (hr, repeating exists)')
    plt.ylabel('Glucose Level')
    plt.legend()
    plt.savefig('trend_plota.png')
    plt.savefig('pretrained/trend_plota.png')
    plt.show()
