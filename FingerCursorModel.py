import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
import numpy as np
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os

name = "small_v6_noShuffle-05"
dataFiles = "data"
fileName = "FingerCursorData"

dataFile = os.path.abspath(f"{dataFiles}/{fileName}.json")
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

target_df = pd.concat([x_result_df["target"], y_result_df["target"]], axis=1)
print(target_df)

combined_df = pd.concat([x_result_df, y_result_df], axis=1)
print(combined_df)
indicies = [np.random.randint(0,combined_df.shape[0]) for _ in range(int(combined_df.shape[0]*0.2))]

test_df = combined_df.loc[indicies]
combined_df.drop(indicies, axis=0, inplace=True)
train_target_df = combined_df[["target"]].copy()
combined_df.drop("target",axis=1, inplace=True)
print(combined_df)

y_test_df = test_df[["target"]].copy()
test_df.drop("target",axis=1, inplace=True)


print('combined_shape:', combined_df.shape)

X_train = combined_df.to_numpy()
y_train = train_target_df.to_numpy()
X_test  = test_df.to_numpy()
y_test  = y_test_df.to_numpy()



# Look under python3 example 2

# exit()
# X_train, X_test, y_train, y_test = train_test_split(feature_tensor, target_tensor, test_size=0.2, random_state=42, shuffle=True)
# final dense 2
# Look to adding drop out layers
# Between dense layers Tell it the fraction of nodes todrop out, between 25-50%
# Weight Decay - an option on the layers
# chapter 5 148-150 drop out and regularization
model = keras.Sequential([
    Dense(84, input_dim=X_train.shape[1], activation='relu'),
    # Dropout(rate=0.25, seed=42),
    Dense(42, activation='relu'),
    # Dropout(rate=0.1, seed=41),
    Dense(21, activation='relu'),

    Dense(2)
])

model.compile(loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=17, batch_size=20, verbose=1, validation_data=(X_test, y_test), shuffle=True)

# history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save(os.path.abspath(f"data/{name}.json"))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# y_pred = model.predict(X_test)




# Make the validation graph
#  Look for overfitting


# For X
# Mean Squared Error (MSE): 77428.81399834645
# Mean Absolute Error (MAE): 195.75334760194184

# For Y
# Mean Squared Error (MSE): 45394.90305828255
# Mean Absolute Error (MAE): 163.08375705185756