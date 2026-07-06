# clip-srt

Telegram bot that downloads a video, transcribes audio locally with Faster Whisper, translates to Persian and returns an MKV with soft subtitles.

## Requirements

- Python 3.10+
- System packages: `ffmpeg`

## Python dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

Copy `.env.example` to `.env` and set `BOT_TOKEN`.

## Usage

1. (Optional) Download a local whisper model:

```bash
python download_model.py
```

2. Run the bot:

```bash
python bot_test.py
```

## Notes & Recommendations

- Add `python-dotenv` to `requirements.txt` (already included).
- Adjust `MAX_CONCURRENT` in `.env` to control concurrency.
- Ensure `ffmpeg` is installed on the host.
- Monitor disk usage in `ARCHIVE_DIR` and implement retention as needed.
