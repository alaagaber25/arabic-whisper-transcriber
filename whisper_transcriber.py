import os
from pathlib import Path

import librosa
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor


class WhisperTranscriber:
    def __init__(self, model_type="egyptian"):
        """
        Initialize Whisper transcriber with specified model type.

        Args:
            model_type: Either "egyptian" or "saudi"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading {model_type} model...")

        if model_type == "egyptian":
            self.processor = WhisperProcessor.from_pretrained(
                "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"
            )
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            self.model = PeftModel.from_pretrained(
                self.model, "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"
            )
        elif model_type == "saudi":
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            self.model = PeftModel.from_pretrained(
                self.model, "Bruno7/whisper-large-v3-turbo-arabic-saudi-phase2"
            )
        else:
            raise ValueError("model_type must be either 'egyptian' or 'saudi'")

        self.model = self.model.to(self.device)
        print(f"{model_type.capitalize()} model loaded successfully!")

    def transcribe(self, audio_path):
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to the WAV file

        Returns:
            Transcription text
        """
        # Load and process audio
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features

        # Cast input features to float16 and move to device
        input_features = input_features.to(self.device, dtype=torch.float16)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, max_length=225)
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

        return transcription

    def unload(self):
        """Unload model from memory."""
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_folder(folder_name, folder_path, model_type, output_dir):
    """
    Process a single folder and save its metadata as CSV.

    Args:
        folder_name: Name of the folder (e.g., "nadia_elsayed")
        folder_path: Full path to the folder
        model_type: "egyptian" or "saudi"
        output_dir: Directory to save individual metadata files

    Returns:
        Number of files processed
    """
    # Check if metadata file already exists
    metadata_file = os.path.join(output_dir, f"{folder_name}_metadata.csv")
    if os.path.exists(metadata_file):
        print(f"\n  Metadata file already exists: {metadata_file}")
        response = input("Skip this folder? (y/n): ").strip().lower()
        if response == "y":
            print(f"Skipping {folder_name}...")
            return 0

    print(f"\n{'=' * 60}")
    print(f"Processing folder: {folder_name}")
    print(f"Using model: {model_type.upper()}")
    print(f"{'=' * 60}")

    # Collect all WAV files
    wav_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    )

    if not wav_files:
        print(f"No WAV files found in {folder_name}")
        return 0

    print(f"Found {len(wav_files)} WAV files")

    # Load model
    transcriber = WhisperTranscriber(model_type=model_type)

    # Process files
    metadata_lines = []
    processed_count = 0

    for wav_file in tqdm(wav_files, desc=f"Transcribing {folder_name}"):
        try:
            file_path = os.path.join(folder_path, wav_file)
            transcription = transcriber.transcribe(file_path)

            # Format: path|transcription|language|speaker
            relative_path = f"{folder_name}/{wav_file}"
            speaker_id = f"speaker_{folder_name}"
            metadata_line = f"{relative_path}|{transcription}|ar|{speaker_id}"
            metadata_lines.append(metadata_line)
            processed_count += 1

        except Exception as e:
            print(f"\n Error processing {wav_file}: {e}")
            continue

    # Unload model to free memory
    transcriber.unload()

    # Save metadata for this folder as CSV
    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_file, "w", encoding="utf-8") as f:
        # Write header
        f.write("audio_path|transcription|language|speaker_id\n")
        # Write data
        f.write("\n".join(metadata_lines))

    print(f"\n Processed {processed_count}/{len(wav_files)} files")
    print(f" Metadata saved to: {metadata_file}")

    return processed_count


def combine_metadata_files(output_dir, folder_names, final_output_file):
    """
    Combine all individual metadata CSV files into one final CSV file.

    Args:
        output_dir: Directory containing individual metadata files
        folder_names: List of folder names
        final_output_file: Path to final combined metadata file
    """
    print(f"\n{'=' * 60}")
    print("Combining metadata files...")
    print(f"{'=' * 60}")

    all_lines = []

    for folder_name in folder_names:
        metadata_file = os.path.join(output_dir, f"{folder_name}_metadata.csv")

        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                # Skip header for individual files (except the first one)
                if not all_lines:
                    # First file: include header
                    all_lines.extend(lines)
                else:
                    # Other files: skip header
                    all_lines.extend(lines[1:])
                print(f" {folder_name}: {len(lines) - 1} entries")
        else:
            print(f"  {folder_name}: metadata file not found")

    # Write combined CSV file
    with open(final_output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

    print(f"\n{'=' * 60}")
    print(f" Combined metadata saved to: {final_output_file}")
    print(f" Total entries: {len(all_lines) - 1}")  # -1 for header
    print(f"{'=' * 60}")


def process_directory(root_dir, output_dir, final_output_file, folder_model_mapping):
    """
    Process all folders and generate metadata files.

    Args:
        root_dir: Root directory containing channel folders
        output_dir: Directory to save individual metadata files
        final_output_file: Path to final combined metadata file
        folder_model_mapping: Dict mapping folder names to model types
    """
    print("Audio Transcription Pipeline")
    print("=" * 60)
    print(f"Input directory: {root_dir}")
    print(f"Metadata directory: {output_dir}")
    print(f"Final output file: {final_output_file}")
    print(f"\nFolder-Model Mapping:")
    for folder, model in folder_model_mapping.items():
        print(f"  {folder}: {model}")
    print("=" * 60)

    total_processed = 0

    # Group folders by model type for efficient processing
    folders_by_model = {"egyptian": [], "saudi": []}

    for folder_name, model_type in folder_model_mapping.items():
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"\n  Warning: Folder not found: {folder_name}")
            continue
        folders_by_model[model_type].append((folder_name, folder_path))

    # Process folders grouped by model type
    for model_type in ["egyptian", "saudi"]:
        folders = folders_by_model[model_type]
        if not folders:
            continue

        print(f"\n{'#' * 60}")
        print(f"# Processing folders with {model_type.upper()} model")
        print(f"{'#' * 60}")

        for folder_name, folder_path in folders:
            count = process_folder(folder_name, folder_path, model_type, output_dir)
            total_processed += count

    # Combine all metadata files
    combine_metadata_files(
        output_dir, list(folder_model_mapping.keys()), final_output_file
    )

    print(f"\n{'=' * 60}")
    print(f" Pipeline Complete!")
    print(f" Total files processed: {total_processed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    ROOT_DIR = r"F:\VOOM-AI\Data\chatterbox\Mics"
    METADATA_DIR = r"F:\VOOM-AI\Data\chatterbox\Mics"
    FINAL_OUTPUT_FILE = r"F:\VOOM-AI\Data\chatterbox\Mics\metadata.csv"

    FOLDER_MODEL_MAPPING = {
        "wavs": "saudi",
    }

    process_directory(ROOT_DIR, METADATA_DIR, FINAL_OUTPUT_FILE, FOLDER_MODEL_MAPPING)
