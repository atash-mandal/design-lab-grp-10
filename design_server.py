from flask import Flask, request 
import numpy as np
import librosa
import keras
import wavio
import os
import pickle
from scipy.io.wavfile import read
from sklearn import preprocessing
import python_speech_features as mfcc
app = Flask(__name__)


# def calculate_delta(array):
#     rows,cols = array.shape
#     deltas = np.zeros((rows,20))
#     N = 2
#     for i in range(rows):
#         index = []
#         j = 1
#         while j <= N:
#             if i-j < 0:
#                 first = 0
#             else:
#                 first = i-j
#             if i+j > rows -1:
#                 second = rows -1
#             else:
#                 second = i+j
#             index.append((second,first))
#             j+=1
#         deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
#     return deltas

# #convert audio to mfcc features
# def extract_features(audio,rate):
#     mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True, nfft=1300)
#     mfcc_feat = preprocessing.scale(mfcc_feat)
#     delta = calculate_delta(mfcc_feat)

#     #combining both mfcc features and delta
#     combined = np.hstack((mfcc_feat,delta))
#     return combined


# def recognize(path): #path is a .wav file
#     print("Entered recognize")
#     modelpath = 'C://Users//atash//OneDrive//Desktop//grp_10//ML//gmm_models'
#     gmm_files = os.listdir(modelpath)
#     print(gmm_files)
#     models = [pickle.load(open('C://Users//atash//OneDrive//Desktop//grp_10//ML//gmm_models//'+fname,'rb')) for fname in gmm_files]
#     print(models)
#     speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
#                 in gmm_files]
#     print("models")
#     sr,audio = read(path)
#     vector = extract_features(audio,sr)
#     log_likelihood = np.zeros(len(models))

#     for i in range(len(models)):
#         gmm = models[i]
#         scores = np.array(gmm.score(vector))
#         log_likelihood[i] = scores.sum()

#     pred = np.argmax(log_likelihood)
#     identity = speakers[pred]

#     if identity == 'unknown':
#         print("Not Recognized! Try again...")

#     else:
#         print( "Recognized as - ", identity)



def extract_mfcc(file_path):
    # Use librosa to extract MFCC (Mel Frequency Cepstral Coefficients) features from audio file
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)


@app.route("/", methods=['POST'])
def authenticate_speaker():
    # Check if the request contains JSON data
    try:
        # Parse JSON data from the request
        data = request.form.getlist('data')

        processed_data = (np.array(data).astype(np.float64)).reshape(-1, 2)

        wavio.write("test.wav", processed_data, 44100, sampwidth=3)

        model = keras.models.load_model("./saved_model.h5")
        # to_predict = []
        # Extract MFCC features from the custom audio file
        custom_mfcc = extract_mfcc("C://Users//atash//OneDrive//Desktop//grp_10//test.wav")

        # Reshape the MFCC features to match the input shape of the model
        custom_mfcc = np.expand_dims(custom_mfcc, axis=0)
        custom_mfcc = np.expand_dims(custom_mfcc, axis=-1)

        # to_predict.append(custom_mfcc)
        predictions = model.predict(custom_mfcc)

        class_names = ["Atash", "Sourish", "Sumajit", "Kowshik"]
        msg = []
        for i, prob in enumerate(predictions[0]):
            msg.append(f"Similarity with {class_names[i]}: {prob*100:.2f}")

        print(predictions)
        return ({"message": msg})
    except:
        return ({"error": "Request must contain JSON data"}), 400

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=5050,debug=True)