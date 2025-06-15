import os
import queue
tmpfile = None
import tempfile

from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from google import genai
import KRTTS

load_dotenv()  # Gemini API key
WHISPER_MODEL_DIR = os.path.expanduser("~/models/whisper")
WHISPER_MODEL_SIZE = "small"
GEMINI_MODEL = "gemini-2.0-flash"
needTTS = False


"""
Запись звука чисто для экзампла
Идея в том, чтобы из сайта брать файл звука, как temp file
Закидывать звук в бэк, обработать текст, выдать в ЛЛМку
"""
def record_audio(path: str, samplerate: int = 16000, channels: int = 1):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"|{status}|", flush=True)
        q.put(indata.copy())

    print("Press Enter to START recording...")
    input()
    print("Recording... press Enter again to STOP.")
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        input()
    print("Stopped recording, saving file...")

    # collect frames
    frames = []
    while not q.empty():
        frames.append(q.get())

    audio = np.concatenate(frames, axis=0)
    sf.write(path, audio, samplerate)
    print(f"Audio saved to {path}\n")


def main():
    print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
    model = whisper.load_model(
        WHISPER_MODEL_SIZE,
        download_root=WHISPER_MODEL_DIR
    )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERR: .env must contain GEMINI_API_KEY")
        return
    client = genai.Client(api_key=api_key)


    """
    У Whisper можно зафорсить какой язык он будет пытаться читать
    Чисто теоретически его можно брать из 'какой язык сайта'
    Но это не особо практично
    """
    lang = input("Force language? [ru / kk / blank=auto]: ").strip().lower() or None
    needTTS = input("Needs TTS? [y/n/blank=no]: ").strip().lower()

    while True:
        cmd = input("[R] to record, [Q] to quit: ").strip().lower()
        if cmd == 'q' or cmd == 'Q':
            print("EXIT")
            break
        if cmd != 'r' or cmd == 'R':
            continue

        wav_path = tempfile.mktemp(suffix='.wav')
        record_audio(wav_path)

        print("Transcribing ->")
        if lang:
            result = model.transcribe(wav_path, language=lang)
        else:
            result = model.transcribe(wav_path)
        transcript = result.get('text', '').strip()
        print(f"> You said: {transcript}\n")
        if not transcript:
            print("ERR: No speech detected. Try again.\n")
            continue


        print("Querying Gemini AI ->")
        # Тут нужно будет помучаться с промпт инжинирингом,
        # чтобы лишние символы не выводило, если TTS всё-таки нужен
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Never use * or other symbols, don't use italics or bold, {transcript}"
        )
        reply = resp.text.strip()
        print(f"LLM REPLY: {reply}\n")
        # Теоретически можно просто очистить reply через regex


        # Русский ТТС мех найти бы получше, может как-то у COQUI взять
        # Казахский я не шарю, но звучит +- норм
        if needTTS == 'y' or needTTS == 'Y':
            print("Triggered TTS YES ->")
            if lang == 'ru':
                print("Triggered RU YES ->")
                KRTTS.synthesize_ru(reply)

                data, samplerate = sf.read('output_temp/output_RU.wav', dtype='float32')
                sd.play(data, samplerate)
                sd.wait()
            elif lang == 'kk':
                print("Triggered KK YES ->")
                KRTTS.synthesize_kk(reply)

                data, samplerate = sf.read('output_temp/output_KZ.wav', dtype='float32')
                sd.play(data, samplerate)
                sd.wait()
            else:
                print("ERR: R U trolling?")
        else:
            pass

if __name__ == '__main__':
    main()
