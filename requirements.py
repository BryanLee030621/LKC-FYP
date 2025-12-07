# Install required packages
pip install librosa soundfile pydub noisereduce numpy
sudo apt-get install ffmpeg  # or brew install ffmpeg on Mac

# If noisereduce gives issues, you can skip it:
# pip install webrtcvad  # Alternative for voice activity detection