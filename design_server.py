from flask import Flask, request 
import numpy as np
import librosa
import keras
import wavio
app = Flask(__name__)

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

        to_predict = []
        to_predict.append(extract_mfcc("C://Users//atash//OneDrive//Desktop//grp_10//test.wav"))
        predictions = model.predict(np.expand_dims(to_predict, axis=-1))

        print(type(predictions))
        return ({"message": predictions.tolist()})
    except:
        return ({"error": "Request must contain JSON data"}), 400

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=5050,debug=True)