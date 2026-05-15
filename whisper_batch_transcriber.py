from pathlib import Path

import librosa
import torch
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor

# Configuration
CHANNELS_TO_PROCESS = [""]
BASE_FOLDER = ""

MODEL_ID = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"
BASE_MODEL_ID = "openai/whisper-large-v3"


# Model loading
def load_model(device: str):
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model = model.to(device)

    return processor, model


# Transcription
def transcribe(wav_file: Path, processor, model, device: str) -> str:
    audio, _ = librosa.load(str(wav_file), sr=16000)
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, dtype=torch.float16)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=225)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def should_skip(wav_file: Path) -> bool:
    if wav_file.with_suffix(".srt").exists():
        print(f"  Skipping {wav_file.name} - .srt already exists")
        return True
    if wav_file.with_suffix(".txt").exists():
        print(f"  Skipping {wav_file.name} - .txt already exists")
        return True
    return False


def process_channel(channel_path: Path, processor, model, device: str):
    wav_files = list(channel_path.rglob("*.wav"))

    if not wav_files:
        print(f"  No .wav files found in {channel_path.name}")
        return

    for wav_file in wav_files:
        if should_skip(wav_file):
            continue

        print(f"  Processing: {wav_file.name}")

        try:
            transcription = transcribe(wav_file, processor, model, device)
            txt_file = wav_file.with_suffix(".txt")
            txt_file.write_text(transcription, encoding="utf-8")
            print(f"  Saved: {txt_file.name}")

        except Exception as e:
            print(f"  Error processing {wav_file.name}: {e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor, model = load_model(device)

    for channel_name in CHANNELS_TO_PROCESS:
        channel_path = Path(BASE_FOLDER) / channel_name

        if not channel_path.exists():
            print(f"Channel folder not found: {channel_path}")
            continue

        print(f"\nProcessing channel: {channel_name}")
        process_channel(channel_path, processor, model, device)

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
