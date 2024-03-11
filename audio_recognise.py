# 
# # Covert Speech into Text 
#

import speech_recognition as sr
import pyttsx3
import os

# Initialize the recognizer 
r = sr.Recognizer() 

# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()

# Function to recognize speech
def recognize_speech():
    try:
        with sr.Microphone() as source2:
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            # listens for the user's input 
            audio = r.listen(source2, timeout=2)
            
            # # Save the recorded audio
            # with open("recorded_audio.wav", "wb") as f:
            #     f.write(audio.get_wav_data())

            # Using google to recognize audio requires internet
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()

            return MyText
                    
    except sr.WaitTimeoutError:
        print("Listening timed out. No phrase detected.")
        return None
        
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None
        
    except sr.UnknownValueError:
        print("Unknown error occurred")
        return None

