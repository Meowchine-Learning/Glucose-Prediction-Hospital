from tcn import TCN, tcn_full_summary
from sklearn.preprocessing import RobustScaler

targets = train_data[['pressure']].to_numpy().reshape(-1, 80)

# drop the unwanted features
train_data.drop(['pressure', 'id', 'breath_id', 'u_out'], axis=1, inplace=True)
test_data =  test_data.drop(['id', 'breath_id', 'u_out'], axis=1)

RS = RobustScaler()
train_data = RS.fit_transform(train_data)
test_data  = RS.transform(test_data)

n_features = train_data.shape[-1]

train_data = train_data.reshape(-1, 80, n_features)
test_data  = test_data.reshape(-1, 80, n_features)

n_epochs = 50
n_splits =  5