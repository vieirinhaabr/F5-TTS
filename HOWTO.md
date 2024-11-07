# FIRST RUN
RUN: python -m venv venv
RUN: pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN: pip install -e .

# HOW TO FORMAT A DATASET
RUN: apt install ffmpeg
RUN: python src\f5_tts\train\datasets\prepare_csv_wavs.py <input> <output>_pinyin --pretrain

# TRAINING 
## SETTINGS ON A40
batch_type = frame
learning_rate = 0,000075
batch_size = 10000 (maybe inscrease on a40)
max_samples = 64
gradient_accumulation = 78
max_gradient_norm = 1
epochs = 100
warmup_updates = 256
save_per_updates = 384
last_per_steps = 5000
precision = fp16
logger = wandb

## AT EVERY START
RUN: source venv/bin/activate

## RUN ON GRADIO (TRAINING)
RUN: apt install ffmpeg
RUN: python src/f5_tts/train/finetune_gradio.py --share