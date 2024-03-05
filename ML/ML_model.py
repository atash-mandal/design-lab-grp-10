import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def extract_mfcc(file_path):
    # Use librosa to extract MFCC (Mel Frequency Cepstral Coefficients) features from audio file
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Assuming you have a folder structure where each person's .wav files are in separate folders.
# Adjust the paths accordingly.

# Step 1: Data Collection and Preprocessing
# per_1 = ['/content/drive/MyDrive/Some Audios-20240130T114552Z-001/Some Audios/User1_1.wav','/content/drive/MyDrive/Some Audios-20240130T114552Z-001/Some Audios/User1_2.wav']
# per_2 = ['/content/drive/MyDrive/Some Audios-20240130T114552Z-001/Some Audios/User2_1.wav','/content/drive/MyDrive/Some Audios-20240130T114552Z-001/Some Audios/User2_2.wav']
# per_3 = ['/content/drive/MyDrive/Some Audios-20240130T114552Z-001/Some Audios/User3_1.wav','/content/drive/MyDrive/Some Audios-20240130T114552Z-001/Some Audios/User3_2.wav']
# per_4 = ['/content/drive/MyDrive/Some Audios-20240130T114552Z-001/Some Audios/User4_1.wav','/content/drive/MyDrive/Some Audios-20240130T114552Z-001/Some Audios/User4_2.wav']

# per_1 = ['/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Atash Audios/Atash1.wav','/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Atash Audios/Atash2.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Atash Audios/Atash3.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Atash Audios/Atash4.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Atash Audios/Atash5.wav']
# per_2 = ['/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Kowshik_audios/kowshik_1.wav','/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Kowshik_audios/kowshik_2.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Kowshik_audios/kowshik_3.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Kowshik_audios/kowshik_4.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Kowshik_audios/kowshik_5.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Kowshik_audios/kowshik_6.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Kowshik_audios/kowshik_7.wav']
# per_3 = ['/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sourish_Audios/Sourish1.wav','/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sourish_Audios/Sourish2.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sourish_Audios/Sourish3.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sourish_Audios/Sourish4.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sourish_Audios/Sourish5.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sourish_Audios/Sourish6.wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sourish_Audios/Sourish7.wav']
# per_4 = ['/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (1).wav','/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (2).wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (3).wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (3).wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (4).wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (5).wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (6).wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (7).wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (8).wav', '/content/drive/MyDrive/Some Audios-20240130T114552Z-001/drive-download-20240131T055825Z-001/Sumajit_audios/Sumajit (9).wav']



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
    data.append(extract_mfcc(path_2 + "//" + file))
    labels.append(1)  # Assuming person2 is labeled as class 1

for file in per_3:
    data.append(extract_mfcc(path_3 + "//" + file))
    labels.append(2)  # Assuming person3 is labeled as class 2

for file in per_4:
    data.append(extract_mfcc(path_4 + "//" + file))
    labels.append(3)  # Assuming person4 is labeled as class 3

data = np.array(data)
labels = np.array(labels)

# Step 2: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Step 3: Train and evaluate a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

rf_predictions = rf_classifier.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy}')

# Step 4: Train and evaluate a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', C=1, random_state=42)
svm_classifier.fit(X_train, y_train)

svm_predictions = svm_classifier.predict(X_val)
svm_accuracy = accuracy_score(y_val, svm_predictions)
print(f'SVM Accuracy: {svm_accuracy}')
