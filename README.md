# clip-srt

A Telegram bot that downloads a video, transcribes its audio locally with Faster Whisper, translates the transcript to Persian, and returns an MKV file with soft subtitles.

[فارسی](README.fa)

## ✨ Overview

clip-srt is a simple local-first workflow for turning a video link into a Persian-subtitled MKV. It is designed for users who want a lightweight, self-hosted experience without relying on cloud transcription services.

## 🚀 Features

- Accepts video URLs sent to a Telegram bot
- Downloads videos with yt-dlp
- Extracts audio and transcribes it locally using Faster Whisper
- Translates subtitles to Persian
- Muxes the subtitles into an MKV container
- Stores processed files in an archive folder and reuses cached results when possible

## 🧰 Tech Stack

- Python 3.10+
- python-telegram-bot
- faster-whisper
- deep-translator
- yt-dlp
- ffmpeg

## 📦 Installation

### 1) Clone the repository

```bash
git clone https://github.com/your-username/clip-srt.git
cd clip-srt
```

### 2) Install system dependencies

Make sure ffmpeg is installed and available in your PATH.

On Ubuntu/Debian:

```bash
sudo apt update && sudo apt install -y ffmpeg
```

### 3) Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Download the local Whisper model

```bash
python download_model.py
```

## ⚙️ Configuration

Copy the example environment file and update it:

```bash
cp .env.example .env
```

Set at least the following values in .env:

- BOT_TOKEN: your Telegram bot token
- PROXY_URL: optional proxy for network requests
- MODEL_PATH: directory for the local Whisper model
- CACHE_DIR: temporary download and processing folder
- ARCHIVE_DIR: folder for completed MKV files
- MODEL_SIZE: Whisper model size, such as base
- MAX_CONCURRENT: number of simultaneous jobs

## ▶️ Usage

Start the bot:

```bash
python bot_test.py
```

Then send a video link to the bot in Telegram. The bot will:

1. Download the video
2. Extract the audio
3. Transcribe it locally
4. Translate the text to Persian
5. Return an MKV with embedded soft subtitles

## 📁 Project Structure

- bot_test.py: Telegram bot entry point and processing pipeline
- download_model.py: downloads the local Whisper model
- requirements.txt: Python dependencies
- archive/: generated MKV files
- cache/: temporary media and subtitle files
- whisper_model/: local model files

## 📝 Notes

- This project is optimized for local processing and does not depend on cloud speech APIs.
- Large files may take longer to process, and the current implementation limits the maximum file size in the download step.
- For long-term use, you may want to add retention policies for archived files and logging.

## 🤝 Contributing

Contributions are welcome. If you improve the bot, fix bugs, or add features, feel free to open a pull request.
