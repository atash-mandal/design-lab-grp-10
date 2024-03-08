import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import os

num_classes = 4

def extract_mfcc(file_path):
    # Use librosa to extract MFCC (Mel Frequency Cepstral Coefficients) features from audio file
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def create_model(input_shape):
    model_ = models.Sequential()

    # Use Conv1D instead of Conv2D
    model_.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model_.add(layers.MaxPooling1D(2))

    model_.add(layers.Flatten())
    model_.add(layers.Dense(64, activation='relu'))
    model_.add(layers.Dense(num_classes, activation='softmax'))

    model_.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_


path_1 = "C://Users//atash//OneDrive//Desktop//grp_10//New folder//Atash_Audios"
path_2 = "C://Users//atash//OneDrive//Desktop//grp_10//New folder//Sourish_Audios"
path_3 = "C://Users//atash//OneDrive//Desktop//grp_10//New folder//Sumajit_Audios"
path_4 = "C://Users//atash//OneDrive//Desktop//grp_10//New folder//Kowshik_Audios"
per_1 = os.listdir(path_1)
per_2 = os.listdir(path_2)
per_3 = os.listdir(path_3)
per_4 = os.listdir(path_4)

data = []
labels = []

for file in per_1:
    data.append(extract_mfcc(path_1 + "//" + file))
    labels.append(0)  # Assuming person1 is labeled as class 0

for file in per_2:
    data.append(extract_mfcc(path_2 + "//" +file))
    labels.append(1)  # Assuming person2 is labeled as class 1

for file in per_3:
    data.append(extract_mfcc(path_3 + "//" +file))
    labels.append(2)  # Assuming person3 is labeled as class 2

for file in per_4:
    data.append(extract_mfcc(path_4 + "//" +file))
    labels.append(3)  # Assuming person4 is labeled as class 3

data = np.array(data)
labels = tf.keras.utils.to_categorical(labels)

# Reshape input data
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

# Step 3: Model Creation
input_shape = X_train[0].shape
model = create_model(input_shape)

# Step 4: Model Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# to_predict = []
# to_predict.append(extract_mfcc(path_4 + "//" +per_4[0]))
# predictions = model.predict(np.expand_dims(to_predict, axis=-1))
# print(predictions)

# Step 5: Model Evaluation
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f'Test accuracy: {test_acc}')


model.save("saved_model.h5")


