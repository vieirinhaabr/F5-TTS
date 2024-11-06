from datasets import load_dataset
from pydub import AudioSegment
import os
import csv

# CONSTANTS
PREPARE_PATH = "E:/Projetos/2000_JOB/9999_AI/F5-TTS/.temp/ptbr"
METADATA_PATH = f"{PREPARE_PATH}/metadata.csv"
WAVS_PATH = f"{PREPARE_PATH}/wavs/"

# LOAD DATASET
ds = load_dataset("mozilla-foundation/common_voice_17_0", "pt", trust_remote_code=True, split='train')

# SAVE ON METADATA ON WRITE   
def save_metadata(audio_file, text, mode="a"):
  with open(METADATA_PATH, mode, newline="", encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter="|")
    writer.writerow([audio_file, text])

# CREATE / SETUP METADATA FILE
if not os.path.exists(METADATA_PATH):
  save_metadata("audio_file", "text", mode="w")

    
# PREPARE DATASET
total_duration = 0
def prepare_dataset(batch):
  path = batch["path"]
  transcription = batch["sentence"]
  
  wav_filename = f"{os.path.basename(os.path.splitext(path)[0])}.wav"
  wav_path = f"{WAVS_PATH}{wav_filename}"
  if path.endswith(".mp3") and not os.path.exists(wav_path):
    audio = AudioSegment.from_mp3(path)
    audio.export(wav_path, format="wav")
    
    global total_duration
    total_duration += audio.duration_seconds
    save_metadata(f"wavs/{wav_filename}", transcription)
    
  else:
    print(f"Skipping: {path}")
  
  return batch

ds.map(prepare_dataset, desc="preprocess dataset")
print(f"Total duration: {total_duration} seconds")