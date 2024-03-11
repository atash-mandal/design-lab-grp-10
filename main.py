import keyboard
from audio_recognise import recognize_speech, SpeakText

# Example usage:
if __name__ == "__main__":
    while True: 
        if keyboard.is_pressed("m"):
            text = recognize_speech()
            if text:
                print("You Said: ", text)
                SpeakText(text)
