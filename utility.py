import wave
import array

def create_wav_file(amplitude_sequence, sample_rate, output_filename):
    
    with wave.open(output_filename, 'w') as wav_file:
        
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample for 16-bit audio
        wav_file.setframerate(sample_rate)

        # Convert amplitude sequence to array of 16-bit integers
        audio_data = array.array('h', [int(amplitude * 32767) for amplitude in amplitude_sequence])

        # Write the audio data to the WAV file
        wav_file.writeframes(audio_data.tobytes())

# Example usage:
amplitude_sequence = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
sample_rate = 44100  # Sample rate in Hz
output_filename = 'output.wav'

create_wav_file(amplitude_sequence, sample_rate, output_filename)
