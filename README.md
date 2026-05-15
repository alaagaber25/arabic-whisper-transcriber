# Arabic Whisper Transcriber

A CUDA-enabled audio transcription pipeline utilizing OpenAI's Whisper `large-v3` models, fine-tuned specifically for Egyptian and Saudi Arabic dialects. 

This project is built to handle local transcription tasks efficiently, supporting both direct text file outputs and metadata CSV generation (ideal for Text-to-Speech dataset preparation).

## Features

*   **Dialect-Specific Models:** Leverages state-of-the-art PEFT fine-tuned models for superior accuracy in:
    *   **Egyptian Arabic:** `AbdelrahmanHassan/whisper-large-v3-egyptian-arabic`
    *   **Saudi Arabic:** `Bruno7/whisper-large-v3-turbo-arabic-saudi-phase2`
*   **Performance:** Optimized for GPU execution utilizing `cuda`, `torch.float16`, and `safetensors`.
*   **Two Modes of Operation:**
    *   **In-Place Transcription (`whisper_transcriber.py`):** Transcribes `.wav` files and saves the output directly adjacent to the audio file as `.txt`. Automatically skips already transcribed files.
    *   **Batch Metadata Generation (`whisper_batch_transcriber.py`):** Transcribes entire directories and aggregates results into a structured `metadata.csv` (format: `audio_path|transcription|language|speaker_id`), which is highly useful for training voice models.

## Prerequisites

*   **Python:** >= 3.11, < 3.12
*   **Hardware:** An NVIDIA GPU is strongly recommended for reasonable transcription speed.

## Installation

This project uses `uv` for fast dependency management as defined in `pyproject.toml`.

1. Ensure `uv` is installed on your system.
2. Install dependencies:
   ```bash
   uv pip install -e .
   ```
   *(This will automatically pull in PyTorch with CUDA 12.4 support on Windows/Linux, along with Transformers, PEFT, Librosa, etc.)*

## Usage

### 1. In-Place Text Transcription (`whisper_transcriber.py`)
This script recursively processes folders and writes `.txt` files next to the source `.wav` files.

**Configuration:**
Before running, open `whisper_transcriber.py` and set your paths at the top of the file:
```python
CHANNELS_TO_PROCESS = ["folder1", "folder2"]
BASE_FOLDER = "C:/path/to/your/audio"
```

**Run:**
```bash
python whisper_transcriber.py
```

### 2. Batch Metadata Generation (`whisper_batch_transcriber.py`)
This script processes folders and generates a compiled `metadata.csv` file mapping audio paths to transcriptions.

**Configuration:**
Open `whisper_batch_transcriber.py` and configure the constants at the bottom of the script:
```python
ROOT_DIR = r"C:\path\to\audio\folders"
METADATA_DIR = r"C:\path\to\output\metadata"
FINAL_OUTPUT_FILE = r"C:\path\to\output\metadata.csv"

# Map folder names to the desired dialect model ("egyptian" or "saudi")
FOLDER_MODEL_MAPPING = {
    "speaker_1_folder": "saudi", 
    "speaker_2_folder": "egyptian"
}
```

**Run:**
```bash
python whisper_batch_transcriber.py
```

## Dependencies
Major libraries used:
*   `transformers` and `peft` for model inference.
*   `torch` (PyTorch) for tensor operations.
*   `librosa` for audio loading and resampling to 16kHz.
*   `tqdm` for progress tracking.