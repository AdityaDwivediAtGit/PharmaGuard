from gtts import gTTS
import os
import tempfile
from utils import time_it, get_logger

logger = get_logger(__name__)

class SpeechHandler:
    def __init__(self):
        # Using a temporary directory for audio files
        self.temp_dir = tempfile.gettempdir()
        logger.info("Speech Handler initialized.")

    @time_it
    def generate_alert(self, text):
        """
        Generates TTS audio file for the given text.
        Returns the path to the generated audio file.
        """
        if not text:
            return None
            
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            # Create a unique filename
            filename = os.path.join(self.temp_dir, f"alert_{hash(text)}.mp3")
            tts.save(filename)
            logger.info(f"Generated speech alert: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to generate speech alert: {e}")
            return None
