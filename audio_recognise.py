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

def recognize_speech_from_wav(wav_file):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    
    # Use the context manager to automatically release the resources
    with sr.AudioFile(wav_file) as source:
        # Adjust the recognizer sensitivity to ambient noise
        # recognizer.adjust_for_ambient_noise(source)
        
        # Listen to the file
        audio_data = recognizer.record(source)
        
        try:
            # Recognize the speech
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        except Exception as e:
            return f"Error: {e}"

def text_to_speech(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Convert text to speech
    engine.say(text)
    
    # Wait for the speech to finish
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

