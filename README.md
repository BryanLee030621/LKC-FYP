#!/bin/bash
# process_all.sh

# Step 1: Preprocess all audio
python preprocessor.py

# Step 2: Generate initial transcriptions (run on Colab with GPU)
echo "Upload the 'preprocess' folder to Google Colab and run:"
echo "python whisper_transcribe.py"

# Step 3: Manual correction
echo "For manual correction, run:"
echo "python correction_tool.py"