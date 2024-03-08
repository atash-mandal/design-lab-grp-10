import keras
import numpy as np
import librosa

def extract_mfcc(file_path):
    # Use librosa to extract MFCC (Mel Frequency Cepstral Coefficients) features from audio file
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)


model = keras.models.load_model("C://Users//atash//OneDrive//Desktop//grp_10//saved_model.h5")
to_predict = []
to_predict.append(extract_mfcc("C://Users//atash//OneDrive//Desktop//grp_10//sourish_1.wav"))
predictions = model.predict(np.expand_dims(to_predict, axis=-1))
print(predictions)

