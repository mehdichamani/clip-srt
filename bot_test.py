import os
import sys
import json
import logging
import asyncio
import tempfile
import portalocker
import threading
import time
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from telegram.warnings import PTBUserWarning
import warnings

warnings.filterwarnings("ignore", category=PTBUserWarning)
load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Network / proxy configuration (do not sweep environment variables)
PROXY_URL = os.getenv("PROXY_URL", "")
# Only export HTTP proxy env vars if the provided PROXY_URL is an HTTP(S) proxy.
# Avoid exporting socks proxies into HTTP_PROXY/HTTPS_PROXY which some libs may not accept.
if PROXY_URL and PROXY_URL.startswith(("http://", "https://")):
    os.environ.setdefault("HTTP_PROXY", PROXY_URL)
    os.environ.setdefault("HTTPS_PROXY", PROXY_URL)

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    logger.error("BOT_TOKEN not found in .env file!")
    sys.exit(1)

from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
import yt_dlp

# Paths & constants
MODEL_PATH = os.getenv("MODEL_PATH", "./whisper_model")
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
ARCHIVE_DIR = os.getenv("ARCHIVE_DIR", "./archive")
HISTORY_FILE = os.getenv("HISTORY_FILE", "./history.json")
LOG_FILE = os.getenv("LOG_FILE", "./service.log")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

def log_action(source, user_info, action, url, file_path="N/A"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [SOURCE: {source}] [USER: {user_info}] [ACTION: {action}] [URL: {url}] [FILE: {file_path}]"
    logger.info(log_line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception:
        logger.exception("Failed to write to log file")

history_lock = threading.Lock()

def load_history():
    history = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    try:
                        portalocker.lock(f, portalocker.LOCK_SH)
                        try:
                            history = json.load(f)
                        except json.JSONDecodeError:
                            history = {}
                    finally:
                        try:
                            portalocker.unlock(f)
                        except Exception:
                            pass
        except Exception:
            logger.exception("Failed to load history file")
    return history

def save_history_atomic(history_obj):
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="history_", dir=os.path.dirname(HISTORY_FILE) or ".")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            try:
                portalocker.lock(f, portalocker.LOCK_EX)
                json.dump(history_obj, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                try:
                    portalocker.unlock(f)
                except Exception:
                    pass
        os.replace(tmp_path, HISTORY_FILE)
    except Exception:
        logger.exception("Failed to write history atomically")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def get_archived_file(url):
    history = load_history()
    return history.get(url)

def save_to_history(url, file_path):
    with history_lock:
        history = load_history()
        history[url] = file_path
        save_history_atomic(history)

def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


async def safe_telegram_call(coro_factory, retries: int = 3, initial_backoff: float = 1.0):
    """Run a coroutine factory that performs a Telegram API call with retries on NetworkError.

    coro_factory should be a zero-arg callable that returns the coroutine to await.
    This ensures resources like files are opened inside the attempt and not reused across retries.
    """
    for attempt in range(1, retries + 1):
        try:
            coro = coro_factory()
            return await coro
        except NetworkError as e:
            logger.warning("Telegram network error (attempt %d/%d): %s", attempt, retries, e)
            if attempt == retries:
                logger.exception("Exceeded Telegram retry attempts")
                raise
            await asyncio.sleep(initial_backoff * (2 ** (attempt - 1)))


def reply_text_factory(message_obj, text):
    async def coro():
        return await message_obj.reply_text(text)
    return coro


def reply_document_factory(message_obj, file_path, caption=None):
    async def coro():
        with open(file_path, 'rb') as f:
            if caption:
                return await message_obj.reply_document(document=f, caption=caption)
            return await message_obj.reply_document(document=f)
    return coro


def edit_text_factory(message_obj, text):
    async def coro():
        return await message_obj.edit_text(text)
    return coro

_MODEL = None
_TRANSLATOR = None
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
PROCESS_SEMAPHORE = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", "2")))

def get_model():
    global _MODEL
    if _MODEL is None:
        # Ensure model directory contains something — do not attempt network download.
        if not os.path.exists(MODEL_PATH) or not any(os.scandir(MODEL_PATH)):
            raise RuntimeError(f"No local model data found in {MODEL_PATH}. Run `python download_model.py` to download the model files before starting the bot.")

        # Temporarily clear proxy env vars to avoid faster-whisper / hf hub trying to use network
        # (and accidentally using non-http proxies such as socks:// which may be rejected).
        proxy_keys = [k for k in list(os.environ.keys()) if 'proxy' in k.lower()]
        saved_proxy = {k: os.environ.pop(k) for k in proxy_keys if k in os.environ}
        try:
            _MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4, download_root=MODEL_PATH)
        except Exception as e:
            logger.exception("Failed to initialize WhisperModel: %s", e)
            raise RuntimeError(f"Failed to initialize WhisperModel: {e}\nEnsure model files are present in {MODEL_PATH} and that no network downloads are required.") from e
        finally:
            # Restore proxy environment
            for k, v in saved_proxy.items():
                os.environ[k] = v
    return _MODEL

def get_translator():
    global _TRANSLATOR
    if _TRANSLATOR is None:
        try:
            _TRANSLATOR = GoogleTranslator(source='auto', target='fa', proxies={"http": PROXY_URL, "https": PROXY_URL})
        except Exception:
            logger.exception("Failed to initialize translator")
            _TRANSLATOR = None
    return _TRANSLATOR

async def process_video(url, user_info):
    base_name = f"video_{int(datetime.now().timestamp())}"
    video_path = os.path.join(CACHE_DIR, f"{base_name}.mp4")
    audio_path = os.path.join(CACHE_DIR, f"{base_name}.wav")
    srt_path = os.path.join(CACHE_DIR, f"{base_name}.srt")
    final_mkv = os.path.join(ARCHIVE_DIR, f"{base_name}.mkv")

    # ۱. دانلود ویدیو با محدودیت‌های معقول
    log_action("Telegram", user_info, "DOWNLOADING", url)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': video_path,
        'proxy': PROXY_URL,
        'noplaylist': True,
        'quiet': True,
        # conservative max size (1 GiB)
        'max_filesize': 1073741824,
        'no_warnings': True,
    }

    # اجرای دانلود ویدیو (blocking lib) in thread
    try:
        await asyncio.to_thread(lambda: yt_dlp.YoutubeDL(ydl_opts).download([url]))
    except Exception:
        logger.exception("yt-dlp failed to download")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"ویدیو دانلود شد اما در مسیر مشخص شده یافت نشد: {video_path}")

    # ۲. استخراج صدای ۱۶کیلوهرتز مونو با استفاده مستقیم از ffmpeg (safer exec)
    log_action("Telegram", user_info, "EXTRACTING AUDIO", url)
    audio_proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await audio_proc.communicate()
    if audio_proc.returncode != 0 or not os.path.exists(audio_path):
        logger.error("ffmpeg audio extraction failed: %s", stderr.decode(errors='ignore') if stderr else "")
        raise FileNotFoundError(f"خطا در استخراج فایل صوتی توسط ffmpeg: {audio_path}")

    # ۳. تشخیص گفتار با استفاده از مدل لوکال Whisper (reused singleton)
    log_action("Telegram", user_info, "TRANSCRIBING", url)
    model = get_model()
    if model is None:
        raise RuntimeError("Whisper model not available")
    segments, _ = await asyncio.to_thread(model.transcribe, audio_path, beam_size=5)
    segments = list(segments)

    # ۴. ترجمه بخش‌ها به فارسی آنلاین (translator singleton)
    log_action("Telegram", user_info, "TRANSLATING", url)
    translator = get_translator()
    if translator is None:
        raise RuntimeError("Translator not available")

    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments, start=1):
            try:
                translated_text = await asyncio.to_thread(translator.translate, segment.text)
            except Exception:
                logger.exception("Translation failed for a segment; falling back to original text")
                translated_text = segment.text
            start_time = format_srt_time(segment.start)
            end_time = format_srt_time(segment.end)
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{translated_text}\n\n")

    # ۵. میکس نهایی (safer exec)
    log_action("Telegram", user_info, "MUXING", url)
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-y', '-i', video_path, '-i', srt_path, '-c', 'copy', '-c:s', 'srt', final_mkv,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.exists(final_mkv):
        logger.error("ffmpeg muxing failed: %s", stderr.decode(errors='ignore') if stderr else "")
        raise RuntimeError("Muxing failed")

    # ۶. پاکسازی فایل‌های موقت کش
    for path in [video_path, audio_path, srt_path]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.exception("Failed to remove temporary file: %s", path)

    save_to_history(url, final_mkv)
    log_action("Telegram", user_info, "COMPLETED", url, final_mkv)
    return final_mkv

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_info = f"@{user.username} (ID: {user.id})" if user.username else f"ID: {user.id}"
    log_action("Telegram", user_info, "COMMAND: /start", "N/A")

    await safe_telegram_call(reply_text_factory(update.message,
        "👋 سلام! به ربات مترجم ویدیو خوش آمدید.\n\n"
        "🔗 کافیست لینک ویدیو یا کلیپ خود را برای من بفرستید تا آن را با زیرنویس فارسی تحویلتان دهم.\n\n"
        "❓ برای اطلاعات بیشتر /help را بزنید."
    ))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reply with a short Persian guide describing available commands and limits."""
    guide = (
        "راهنما:\n\n"
        "- ارسال لینک: یک لینک ویدیویی برای من بفرست؛ ربات ویدیو را دانلود، صوت را استخراج، متن را تشخیص و ترجمه می‌کنه و فیلم با زیرنویس رو برات میفرسته.\n\n"
        "- فرمان‌ها: /start برای خوش‌آمدگویی، /help برای نمایش این راهنما.\n\n"
        "- محدودیت‌ها: کلیپ های بالای یک گیگ رو نمیتونیم پردازش کنیم فعلا،  تعداد پردازش همزمان کلیپ ها محدودع، پس اگه طول میکشه یکم صبر کن.\n\n"
        "- پشتیبانی: فعلا ربات در حال توسعه است پس اگ خطا داد واسم بفرستید به این آیدی @mehdi_chamani تا درستش کنم."
    )

    await safe_telegram_call(reply_text_factory(update.message, guide))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received message: %s", update.message.text)

    url = (update.message.text or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        logger.warning("Ignored non-URL message")
        return

    user = update.message.from_user
    user_info = f"@{user.username} (ID: {user.id})" if user.username else f"ID: {user.id}"

    log_action("Telegram", user_info, "RECEIVED", url)

    cached_file = get_archived_file(url)
    if cached_file and os.path.exists(cached_file):
        log_action("Telegram", user_info, "CACHE_HIT", url, cached_file)
        await safe_telegram_call(reply_text_factory(update.message, "✨ این لینک قبلاً پردازش شده است! در حال ارسال فایل اصلی آرشیو..."))
        await safe_telegram_call(reply_document_factory(update.message, cached_file))
        return

    status_msg = await safe_telegram_call(reply_text_factory(update.message, "⏳ لینک شما به صف پردازش اضافه شد. در حال دانلود و آماده‌سازی زیرنویس فارسی..."))

    try:
        async with PROCESS_SEMAPHORE:
            output_mkv = await process_video(url, user_info)
        await safe_telegram_call(edit_text_factory(status_msg, "📤 فرآیند ترجمه به پایان رسید! در حال ارسال فایل ویدیو (Uncompressed MKV)..."))
        await safe_telegram_call(reply_document_factory(update.message, output_mkv))
    except Exception as e:
        log_action("Telegram", user_info, "FAILED", url)
        logger.exception("Processing failed for %s", url)
        await safe_telegram_call(reply_text_factory(update.message, f"❌ خطایی در طول فرآیند پردازش رخ داد: {str(e)}"))

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception: %s", context.error)


MAX_RESTART_DELAY = 60


def initialize_app():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .proxy(PROXY_URL)
        .get_updates_proxy(PROXY_URL)
        .connect_timeout(30)
        .read_timeout(120)
        .write_timeout(120)
        .pool_timeout(60)
        .connection_pool_size(8)
        .get_updates_connect_timeout(30)
        .get_updates_read_timeout(30)
        .get_updates_write_timeout(30)
        .get_updates_pool_timeout(30)
        .get_updates_connection_pool_size(1)
        .build()
    )
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    return app


def main():
    logger.info("🚀 Local pipeline bot testing script initialized...")
    logger.info("🌐 Proxy bridge: %s", PROXY_URL)
    logger.info("🔋 Listening for message links incoming via Telegram...")

    delay = 1
    while True:
        try:
            app = initialize_app()
            app.run_polling()
            delay = 1
        except Exception:
            logger.exception("Polling stopped unexpectedly, restarting in %ds...", delay)
            time.sleep(delay)
            delay = min(delay * 2, MAX_RESTART_DELAY)


if __name__ == "__main__":
    main()