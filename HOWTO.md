## FIRST RUN
RUN: python -m venv venv

## AT FIRST TIME
RUN: pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN: pip install -e .

## TO PREPARE DATASET
RUN: apt install ffmpeg
RUN: python src\f5_tts\train\datasets\prepare_csv_wavs.py <input> <output>_pinyin --pretrain

## AT EVERY START
RUN: source venv/bin/activate

## RUN ON GRADIO (TRAINING)
RUN: apt install ffmpeg
RUN: python src/f5_tts/train/finetune_gradio.py --share