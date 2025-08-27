#!/usr/bin/env python3
"""
Voice + RAG Chatbot Integration with ElevenLabs TTS
Full voice conversation: Speech-to-Text → RAG Chatbot → Text-to-Speech
"""

import argparse
import queue
import re
import sys
import time
import uuid
import os
import io
import threading
from threading import Event
from io import StringIO
import contextlib

from google.cloud import speech
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play

# Import your existing chatbot
from memoryrag import stream_graph_updates, graph

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)


class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech handler using v3 Turbo model."""
    
    def __init__(self, api_key: str, voice_id: str = None, model: str = "eleven_turbo_v2_5"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # Default to a good Arabic voice if none provided
        self.voice_id = voice_id or "pNInz6obpgDQGcFmaJgB"  # Adam (multilingual)
        
        # Verify API key and get available voices
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify API connection and list available voices."""
        headers = {"xi-api-key": self.api_key}
        
        try:
            response = requests.get(f"{self.base_url}/voices", headers=headers)
            if response.status_code == 200:
                voices = response.json()["voices"]
                print(f"✅ ElevenLabs connected. Available voices: {len(voices)}")
                
                # Show some good multilingual voices
                multilingual_voices = [v for v in voices if "multilingual" in v.get("labels", {}).get("use_case", "").lower()]
                if multilingual_voices:
                    print("🎙️ Recommended multilingual voices:")
                    for voice in multilingual_voices[:3]:
                        print(f"   - {voice['name']} (ID: {voice['voice_id']})")
            else:
                print(f"⚠️ ElevenLabs API issue: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ ElevenLabs connection error: {e}")
    
    def speak(self, text: str, play_audio: bool = True) -> bool:
        """Convert text to speech and optionally play it."""
        if not text.strip():
            return False
            
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            print("🔊 Generating speech...")
            response = requests.post(
                f"{self.base_url}/text-to-speech/{self.voice_id}",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                if play_audio:
                    # Play audio directly from memory
                    audio_data = io.BytesIO(response.content)
                    audio = AudioSegment.from_mp3(audio_data)
                    
                    # Play in a separate thread to avoid blocking
                    def play_thread():
                        try:
                            print(">>> Playback thread: Attempting to play audio...")
                            play(audio)
                            print(">>> Playback thread: Audio finished.")
                        except Exception as e:
                            print(f"Audio playback error: {e}")
                    
                    thread = threading.Thread(target=play_thread)
                    thread.daemon = True
                    thread.start()
                    
                return True
            else:
                print(f"TTS Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"TTS Error: {e}")
            return False


class ChatbotProcessor:
    """Handles chatbot interactions with session management."""
    
    def __init__(self, tts_enabled: bool = False, tts_handler: ElevenLabsTTS = None):
        self.session_id = str(uuid.uuid4())
        self.thread_id = "1"
        self.config = {"configurable": {"thread_id": f"{self.session_id}-{self.thread_id}"}}
        self.tts_enabled = tts_enabled
        self.tts_handler = tts_handler
        print(f"🤖 Chatbot session initialized: {self.session_id}")
    
    def process_query(self, query: str) -> str:
        """Process query through your RAG chatbot and capture the response."""
        if not query.strip():
            return ""
        
        # Capture the printed output from your chatbot
        captured_output = StringIO()
        
        try:
            # Redirect stdout to capture the assistant's response
            with contextlib.redirect_stdout(captured_output):
                stream_graph_updates(query, self.config)
            
            # Get the captured output
            response = captured_output.getvalue()
            
            # Extract just the assistant's response (remove "Assistant: " prefix)
            if response.startswith("Assistant: "):
                response = response[11:].strip()
            elif "Assistant:" in response:
                # Handle cases where there might be other text before "Assistant:"
                parts = response.split("Assistant:")
                if len(parts) > 1:
                    response = parts[-1].strip()
            
            # Speak the response if TTS is enabled
            if self.tts_enabled and self.tts_handler and response:
                self.tts_handler.speak(response)
            
            return response
            
        except Exception as e:
            error_msg = f"خطأ في معالجة الاستعلام: {str(e)}"
            if self.tts_enabled and self.tts_handler:
                self.tts_handler.speak(error_msg)
            return error_msg
    
    def new_conversation(self):
        """Start a new conversation thread."""
        self.thread_id = str(int(self.thread_id) + 1)
        self.config = {"configurable": {"thread_id": f"{self.session_id}-{self.thread_id}"}}
        print(f"🔄 New conversation started: Thread {self.thread_id}")
        
        # Announce new conversation via TTS if enabled
        if self.tts_enabled and self.tts_handler:
            self.tts_handler.speak("محادثة جديدة بدأت. كيف يمكنني مساعدتك؟")


class MicrophoneStream:
    """Audio stream handler."""
    
    def __init__(self, rate: int = RATE, chunk: int = CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        self._audio_interface = None
        self._audio_stream = None

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def voice_rag_loop(responses, stream, stop_event, chatbot, args):
    """Main loop that processes speech and sends to RAG chatbot."""
    num_chars_printed = 0
    
    for response in responses:
        if stop_event.is_set():
            break
            
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript.strip()
        if not transcript:
            continue

        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            # Show interim results
            sys.stdout.write(f"🎤 {transcript}" + overwrite_chars + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript) + 3
        else:
            # Final result - process with chatbot
            print(f"🎤 {transcript}" + overwrite_chars)
            
            # Check for special commands
            if re.search(r"\b(exit|quit|stop|خروج|توقف|إيقاف)\b", transcript, re.I):
                print("وداعاً... / Goodbye!")
                if chatbot.tts_enabled and chatbot.tts_handler:
                    chatbot.tts_handler.speak("وداعاً!")
                stop_event.set()
                break
            elif re.search(r"\b(new|جديد|محادثة جديدة)\b", transcript, re.I):
                chatbot.new_conversation()
                num_chars_printed = 0
                continue
            
            # Process query with RAG chatbot
            print("🤔 Processing...")
            bot_response = chatbot.process_query(transcript)
            
            if bot_response:
                print(f"🤖 Assistant: {bot_response}")
            else:
                fallback_msg = "عذراً، لم أتمكن من معالجة استفسارك."
                print(f"🤖 Assistant: {fallback_msg}")
                if chatbot.tts_enabled and chatbot.tts_handler:
                    chatbot.tts_handler.speak(fallback_msg)
                
            num_chars_printed = 0
            print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language_code",
        default="ar-SA",
        help="Language code for recognition (default: ar-SA)"
    )
    parser.add_argument(
        "--model",
        default="latest_long",
        help="Speech recognition model (default: latest_long)"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced model for better accuracy"
    )
    parser.add_argument(
        "--elevenlabs_key",
        default=None,
        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)"
    )
    parser.add_argument(
        "--voice_id",
        default="pNInz6obpgDQGcFmaJgB",  # Adam (multilingual)
        help="ElevenLabs voice ID (default: Adam)"
    )
    parser.add_argument(
        "--tts_model",
        default="eleven_turbo_v2_5",
        help="ElevenLabs model (default: eleven_turbo_v2_5 - latest fast model)"
    )
    parser.add_argument(
        "--no_tts",
        action="store_true",
        help="Disable text-to-speech output"
    )
    
    args = parser.parse_args()

    # Get ElevenLabs API key
    api_key = args.elevenlabs_key or os.getenv('ELEVENLABS_API_KEY') #big mistake but let it be for now (debugging purposes)
    
    # Initialize TTS if enabled and API key provided
    tts_handler = None
    tts_enabled = not args.no_tts and api_key
    
    if tts_enabled:
        if not api_key:
            print("⚠️ ElevenLabs API key not provided. TTS disabled.")
            print("   Set --elevenlabs_key or ELEVENLABS_API_KEY environment variable")
            tts_enabled = False
        else:
            try:
                tts_handler = ElevenLabsTTS(api_key, args.voice_id, args.tts_model)
                print(f"🔊 TTS enabled with voice: {args.voice_id}")
            except Exception as e:
                print(f"⚠️ TTS initialization failed: {e}")
                tts_enabled = False

    # Initialize chatbot processor
    chatbot = ChatbotProcessor(tts_enabled, tts_handler)
    
    # Initialize Speech client
    client = speech.SpeechClient()
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=args.language_code,
        model=args.model,
        use_enhanced=args.enhanced,
        enable_automatic_punctuation=True,
        enable_spoken_punctuation=True,
        enable_spoken_emojis=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False,
    )

    stop_event = Event()
    
    print("🎙️🤖🔊 Full Voice Conversational AI")
    print("=" * 50)
    print(f"🌍 Language: {args.language_code}")
    print(f"📚 RAG Knowledge Base: Loaded")
    print(f"🧠 Model: Qwen2.5-32B with Tools")
    print(f"🔊 TTS: {'ElevenLabs ' + args.tts_model if tts_enabled else 'Disabled'}")
    print()
    print("Voice Commands:")
    print("  - 'خروج' or 'exit' → Exit")
    print("  - 'جديد' or 'new' → New conversation")
    print()
    print("ابدأ بطرح أسئلتك بصوتك... (Start asking questions with your voice)")
    print("=" * 50)
    
    # Welcome message
    if tts_enabled and tts_handler:
        tts_handler.speak("مرحباً! أنا مساعدك الذكي. كيف يمكنني مساعدتك اليوم؟")

    try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)
            voice_rag_loop(responses, stream, stop_event, chatbot, args)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        if tts_enabled and tts_handler:
            tts_handler.speak("وداعاً!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure:")
        print("1. Your vLLM server is running")
        print("2. ChromaDB and knowledge base are set up")
        print("3. Google Cloud Speech API is configured")
        print("4. ElevenLabs API key is valid")


if __name__ == "__main__":
    main()