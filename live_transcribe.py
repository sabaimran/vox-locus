import pyaudio # type: ignore
import numpy as np
import threading
import time
import wave
import os
from datetime import datetime
import tempfile
from faster_whisper import WhisperModel # type: ignore

class LiveTranscriber:
    def __init__(self, model_size="base", device="cpu"):
        """
        Initialize the live transcriber with the specified Whisper model size.
        
        Args:
            model_size (str): Size of the Whisper model to use 
                             ("tiny", "base", "small", "medium", "large")
            device (str): Device to run the model on ("cpu" or "cuda")
        """
        print(f"Loading Faster Whisper model ({model_size})...")
        self.model = WhisperModel(model_size, device=device)
        print("Model loaded!")
        
        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper expects 16kHz audio
        self.chunk = 1024
        self.record_seconds = 5  # Process audio in 5-second chunks
        
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.transcription_thread = None
        
        self.start_time = time.time()
        self.all_frames = []
        self.all_transcriptions = []
        
    def start_recording(self):
        """Start recording audio from the microphone."""
        self.is_recording = True
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("Recording started. Speak into the microphone...")
        
        # Start a separate thread for continuous transcription
        self.transcription_thread = threading.Thread(target=self.transcribe_continuously)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
    def transcribe_continuously(self):
        """Continuously record and transcribe audio in chunks."""
        while self.is_recording:
            # Clear previous frames
            self.frames = []
            
            # Record audio for the specified duration
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not self.is_recording:
                    break
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
                
            self.all_frames.extend(self.frames)
            
            if self.frames and self.is_recording:
                # Save the recorded audio to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                self.save_audio(temp_file.name, self.frames)
                
                # Transcribe the audio
                segments, _ = self.model.transcribe(temp_file.name, beam_size=5)
                transcription = " ".join([segment.text for segment in segments])
                
                if transcription:
                    elapsed_time = time.time() - self.start_time
                    print(f"{elapsed_time:.2f}s: {transcription}")
                    
                self.all_transcriptions.append(transcription)
                
                # Clean up the temporary file
                try:
                    temp_file.close()
                except Exception as e:
                    print(f"Error closing temp file: {e}")
    
    def save_audio(self, filename, frames):
        """Save the recorded audio frames to a WAV file."""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def stop_recording(self):
        """Stop the recording process."""
        self.is_recording = False
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        
        if self.transcription_thread:
            self.transcription_thread.join(timeout=1)
        
        print("Recording stopped.")
    
    def close(self):
        """Clean up resources."""
        self.stop_recording()
        self.audio.terminate()
        
        folder_name = f"transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(folder_name, exist_ok=True)
        output_filename = os.path.join(folder_name, f"complete_audio.wav")
        if self.all_frames:
            self.save_audio(output_filename, self.all_frames)
            
        incremental_transcription = os.path.join(folder_name, f"incremental_transcription.txt")
        with open(incremental_transcription, 'w') as f:
            for transcription in self.all_transcriptions:
                f.write(transcription + "\n")
        print(f"Transcriptions saved to {incremental_transcription}")
        print(f"Audio saved to {output_filename}")
        
        full_transcription = ""
        
        full_transcription_output_path = os.path.join(folder_name, f"full_transcription.txt")
        segments, _ = self.model.transcribe(output_filename, beam_size=5)
        for segment in segments:
            full_transcription += segment.text + " "
        full_transcription = full_transcription.strip()
        
        with open(full_transcription_output_path, 'w') as f:
            f.write(full_transcription)
        
        print(f"Full transcription saved to {full_transcription_output_path}")
        print(f"Full transcription: {full_transcription}")
        print("Transcriber closed.")


def main():
    # Create a transcriber with the "base" model
    transcriber = LiveTranscriber(model_size="base", device="cpu")
    
    try:
        # Start recording and transcribing
        transcriber.start_recording()
        
        # Keep the program running until Ctrl+C is pressed
        print("Press Ctrl+C to stop recording")
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Clean up
        transcriber.close()


if __name__ == "__main__":
    main()