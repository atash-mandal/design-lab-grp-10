import keyboard
import requests
import json
import sounddevice as sd

if __name__ == "__main__":
    print("main")
    while True:
        if keyboard.is_pressed("m"):
            fs=44100
            duration = 2  # seconds
            print("Recording Audio for "+str(duration)+" seconds")
            myrecording = sd.rec(duration * fs, samplerate=fs, channels=2, dtype='float64')
            sd.wait()
            print("Audio recording complete , Play Audio")
            # print(myrecording)
            base_url = "http://192.168.29.184:5050"
            data = [1,2,3,4,5]
            print(myrecording)
            
            response = requests.post(base_url, data={
                "data": myrecording.tolist()
            })
            if response.status_code == 200:
                print("Request successful!")
                print("Response:", response.json())
                # print(response.json()["message"])
            else:
                print("Request failed with status code:", response.status_code)
                print("Response:", response.text)
            
