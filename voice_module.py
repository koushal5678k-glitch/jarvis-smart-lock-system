import speech_recognition as sr
import datetime
import threading
import queue
import time

class VoiceModule:
    def __init__(self):
        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Audio Queue for non-blocking Text-to-Speech
        self.tts_queue = queue.Queue()
        
        # Start a dedicated thread for Text-To-Speech
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        # Adjust for ambient noise once at startup
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def _tts_worker(self):
        """
        Dedicated worker thread that processes text-to-speech without freezing the main application.
        We bypassed pyttsx3 and are using win32com directly, which fixes the 'silent audio' bug 
        common on Windows by directly interfacing with SAPI5.
        """
        import pythoncom
        # Must be called for COM instances in new threads on Windows
        pythoncom.CoInitialize()
        import win32com.client
        
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Rate = 2  # Set talk speed (range -10 to 10)
        speaker.Volume = 100

        while True:
            text = self.tts_queue.get()
            if text is None:  # Shutdown signal
                break
                
            # Perform direct TTS
            speaker.Speak(text)
            
            self.tts_queue.task_done()

    def speak(self, text):
        """Speak the given text synchronously (wait until finished)."""
        print(f"Jarvis AI: {text}")
        self.tts_queue.put(text)
        self.tts_queue.join()  # block until the queue completes the speaking task

    def speak_async(self, text):
        """Speak the given text without blocking the main thread."""
        print(f"Jarvis AI: {text}")
        self.tts_queue.put(text)

    def get_time_greeting(self):
        """Smart add-on: time-based greeting."""
        hour = datetime.datetime.now().hour
        if 0 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 18:
            return "Good afternoon"
        else:
            return "Good evening"

    def listen_for_wake_word(self):
        """Continuously listen and return True when 'hey jarvis' is detected."""
        print("Listening for wake word 'Hey Jarvis'...")
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Heard: {command}")
                if "jarvis" in command or "hey jarvis" in command:
                    return True
            except sr.WaitTimeoutError:
                pass # Timeout expected
            except sr.UnknownValueError:
                pass # Unrecognizable speech
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
        return False

    def listen_for_command(self):
        """Listen for the user's explicit command after wake word activation."""
        print("Listening for command...")
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Command heard: {command}")
                return command
            except sr.WaitTimeoutError:
                print("No command heard (Timeout).")
            except sr.UnknownValueError:
                print("Could not understand command.")
            except sr.RequestError as e:
                print(f"Speech recognition service error; {e}")
        return None
