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

def split_coordinates(df):
    x_data = {}
    y_data = {} 
    for col in df.columns:
        x_data[col] = df[col].apply(lambda x: x[2])  # Extract x-coordinate
        y_data[col] = df[col].apply(lambda x: x[3])  # Extract y-coordinate
    x_df = pd.DataFrame(x_data)
    y_df = pd.DataFrame(y_data)
    # print("split x_data:",x_df)

    return x_df, y_df

name = "small_v2_gesture"
dataFiles = "gesture"
fileName = "GestureData"
handUsed = '0'
otherHand = '1'
targetMap = {'idle':0, 'lclick':1, 'rclick':2}

dataFile = os.path.abspath(f"data/{dataFiles}/{fileName}.json")
with open(dataFile, 'r') as j:
     data = json.loads(j.read())


# Creating feature tensor
df = pd.DataFrame.from_dict(data[handUsed])
# df.drop(range(0,11),inplace=True)
# print(df.head())
x_result_df, y_result_df = split_coordinates(df)

combined_df = pd.concat([x_result_df, y_result_df], axis=1)
print(combined_df)

# feature_tensor = combined_df.to_numpy()

# Creating target tensor


for i in range(len(data["dot"])):

    data["dot"][i] = targetMap[data["dot"][i]]
# df.drop(range(0,11),inplace=True)
df = pd.DataFrame.from_dict(data["dot"])
print(df.head())


indicies = [np.random.randint(0,combined_df.shape[0]) for _ in range(int(combined_df.shape[0]*0.2))]

test_df_x = combined_df.loc[indicies]
combined_df.drop(indicies,axis=0,inplace=True)
test_df_y = df.loc[indicies]
df.drop(indicies,axis=0,inplace=True)
print(test_df_y.head())

# combined_df.drop(indicies, axis=0, inplace=True)
# # train_target_df = combined_df[["target"]].copy()
# combined_df.drop("target",axis=1, inplace=True)
# print(combined_df)

# y_test_df = test_df[["target"]].copy()
# test_df.drop("target",axis=1, inplace=True)


# print('combined_shape:', combined_df.shape)
# # combined_df = combined_df.sample(frac=1)

X_train = combined_df.to_numpy()
y_train = df.to_numpy()
X_test  = test_df_x.to_numpy()
y_test  = test_df_y.to_numpy()



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
    # Dropout(rate=0.25),
    Dense(42, activation='relu'),
    # Dropout(rate=0.1, seed=41),
    Dense(21, activation='relu'),

    Dense(3, activation='softmax')
])


#  validation_data=(X_test, y_test)
model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=300, batch_size=20, validation_data=(X_test, y_test), verbose=1, shuffle=True)
model.save(os.path.abspath(f"models/{name}.keras"))

# history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# y_pred = model.predict(X_test)
