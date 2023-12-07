import tensorflow as tf
import keras
from keras.layers import Dense
import numpy as np
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataFile = "/Users/jackcameback/Classes/Fall2023/MachineLearning/HandTrackingProject/data/FingerCursorData.json"
trainX = [] # train Data
df = pd.read_json(
    dataFile,
)
# print(df)
df.drop(["1"], axis=1, inplace=True)
dataFrame = pd.DataFrame()
tempDict = {}

for i in range(len(df['0'])):

    for layer in df["0"][i]:
        try:
            tempDict[layer[1]].append((layer[2],layer[3]))
        except:
            tempDict[layer[1]] = [(layer[2],layer[3])]
    try:
        tempDict["target"].append(df["dot"][i])
    except:
        tempDict["target"] = [df["dot"][i]]

            
dataFrame = pd.DataFrame.from_dict(tempDict)

print(dataFrame)


# Function to split x, y coordinates
def split_coordinates(df):
    x_data = {}
    y_data = {}
    for col in df.columns:
        x_data[col] = df[col].apply(lambda x: x[0])  # Extract x-coordinate
        y_data[col] = df[col].apply(lambda x: x[1])  # Extract y-coordinate

    x_df = pd.DataFrame(x_data)
    y_df = pd.DataFrame(y_data)

    return x_df, y_df

x_result_df, y_result_df = split_coordinates(dataFrame)
print("X DataFrame:")
print(x_result_df)
print("\nY DataFrame:")
print(y_result_df)
# target_df = pd.concat([x_result_df["target"], y_result_df["target"]], axis=1)
# print(target_df)
# combined_df = pd.concat([x_result_df, y_result_df], axis=1)
# print(combined_df)
# combined_df.drop("target",axis=1, inplace=True)
# print(combined_df)
x_target = pd.Series(x_result_df["target"])
x_result_df.drop(["target"], axis=1, inplace=True)
print(x_target)
y_target = pd.Series(y_result_df["target"])
y_result_df.drop(["target"], axis=1, inplace=True)
print(y_target)

X_train, X_test, y_train, y_test = train_test_split(x_result_df, x_target, test_size=0.2, random_state=42)

model = keras.Sequential([
    Dense(84, input_dim=X_train.shape[1], activation='relu'),
    Dense(42, activation='relu'),
    Dense(21, activation='relu'),
    Dense(1)
])

model.compile(loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
model.save('/Users/jackcameback/Classes/Fall2023/MachineLearning/HandTrackingProject/models/x_predict.keras')
# For X
# Mean Squared Error (MSE): 77428.81399834645
# Mean Absolute Error (MAE): 195.75334760194184

# For Y
# Mean Squared Error (MSE): 45394.90305828255
# Mean Absolute Error (MAE): 163.08375705185756