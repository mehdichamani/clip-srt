import os
import sys
import json
import logging
import asyncio
import tempfile
import portalocker
import threading
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import NetworkError
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
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

# Max characters per Telegram message
TELEGRAM_MAX_CHARS = 4000

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


def reply_markup_factory(message_obj, text, reply_markup):
    async def coro():
        return await message_obj.reply_text(text, reply_markup=reply_markup)
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


# ---------------------------------------------------------------------------
# Phase 1: download, extract audio, transcribe, translate → return raw data
# ---------------------------------------------------------------------------

async def prepare_video(url: str, user_info: str) -> dict:
    """Download video, transcribe and translate audio.

    Returns a dict with keys:
        video_path  – path to downloaded MP4
        srt_path    – path to generated SRT file
        segments    – list of dicts {original, translated, start, end}
    The caller is responsible for either muxing or discarding video_path.
    """
    base_name = f"video_{int(datetime.now().timestamp())}"
    video_path = os.path.join(CACHE_DIR, f"{base_name}.mp4")
    audio_path = os.path.join(CACHE_DIR, f"{base_name}.wav")
    srt_path = os.path.join(CACHE_DIR, f"{base_name}.srt")

    # 1. Download video with reasonable constraints
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

    # Run blocking yt-dlp download in thread pool
    try:
        await asyncio.to_thread(lambda: yt_dlp.YoutubeDL(ydl_opts).download([url]))
    except Exception:
        logger.exception("yt-dlp failed to download")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video downloaded but not found at expected path: {video_path}")

    # 2. Extract 16 kHz mono audio via ffmpeg (safer exec)
    log_action("Telegram", user_info, "EXTRACTING AUDIO", url)
    audio_proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await audio_proc.communicate()
    if audio_proc.returncode != 0 or not os.path.exists(audio_path):
        logger.error("ffmpeg audio extraction failed: %s", stderr.decode(errors='ignore') if stderr else "")
        raise FileNotFoundError(f"ffmpeg failed to extract audio: {audio_path}")

    # 3. Speech recognition with local Whisper model (singleton)
    log_action("Telegram", user_info, "TRANSCRIBING", url)
    model = get_model()
    if model is None:
        raise RuntimeError("Whisper model not available")
    raw_segments, _ = await asyncio.to_thread(model.transcribe, audio_path, beam_size=5)
    raw_segments = list(raw_segments)

    # 4. Translate segments to Persian online (translator singleton)
    log_action("Telegram", user_info, "TRANSLATING", url)
    translator = get_translator()
    if translator is None:
        raise RuntimeError("Translator not available")

    segments = []
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, seg in enumerate(raw_segments, start=1):
            try:
                translated_text = await asyncio.to_thread(translator.translate, seg.text)
            except Exception:
                logger.exception("Translation failed for segment %d; falling back to original", i)
                translated_text = seg.text
            start_time = format_srt_time(seg.start)
            end_time = format_srt_time(seg.end)
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{translated_text}\n\n")
            segments.append({
                "original": seg.text.strip(),
                "translated": translated_text.strip() if translated_text else seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
            })

    # Remove intermediate audio file — no longer needed
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        logger.exception("Failed to remove audio file: %s", audio_path)

    return {
        "video_path": video_path,
        "srt_path": srt_path,
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Phase 2a: mux clip with subtitles
# ---------------------------------------------------------------------------

async def mux_clip(video_path: str, srt_path: str, url: str, user_info: str) -> str:
    """Mux the video with the SRT subtitle and return the final MKV path."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    final_mkv = os.path.join(ARCHIVE_DIR, f"{base_name}.mkv")

    log_action("Telegram", user_info, "MUXING", url)
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-y', '-i', video_path, '-i', srt_path, '-c', 'copy', '-c:s', 'srt', final_mkv,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.exists(final_mkv):
        logger.error("ffmpeg muxing failed: %s", stderr.decode(errors='ignore') if stderr else "")
        raise RuntimeError("Muxing failed")

    # Clean up source files
    for path in [video_path, srt_path]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.exception("Failed to remove temporary file: %s", path)

    save_to_history(url, final_mkv)
    log_action("Telegram", user_info, "COMPLETED (CLIP)", url, final_mkv)
    return final_mkv


# ---------------------------------------------------------------------------
# Phase 2b: build dual-language text output, skip mux entirely
# ---------------------------------------------------------------------------

def build_dual_text(segments: list) -> list[str]:
    """Build dual-language (original + Persian) text messages split to fit Telegram limits."""
    lines = []
    for seg in segments:
        lines.append(seg["original"])
        lines.append(seg["translated"])
        lines.append("")  # blank separator between pairs

    # Split into chunks that fit within Telegram's character limit
    chunks: list[str] = []
    current_chunk = ""
    for line in lines:
        candidate = current_chunk + line + "\n"
        if len(candidate) > TELEGRAM_MAX_CHARS and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk = candidate
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def cleanup_prepare_files(video_path: str, srt_path: str):
    """Remove temporary files when the user chose text-only output."""
    for path in [video_path, srt_path]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.exception("Failed to remove temporary file: %s", path)


# ---------------------------------------------------------------------------
# Pending job store: maps job_id → job payload stored in bot_data
# ---------------------------------------------------------------------------

PENDING_JOBS_KEY = "pending_jobs"


def store_pending_job(bot_data: dict, job_id: str, payload: dict):
    if PENDING_JOBS_KEY not in bot_data:
        bot_data[PENDING_JOBS_KEY] = {}
    bot_data[PENDING_JOBS_KEY][job_id] = payload
    logger.info("Stored pending job %s", job_id)


def pop_pending_job(bot_data: dict, job_id: str) -> dict | None:
    jobs = bot_data.get(PENDING_JOBS_KEY, {})
    job = jobs.pop(job_id, None)
    if job:
        logger.info("Popped pending job %s", job_id)
    return job


# ---------------------------------------------------------------------------
# Telegram command handlers
# ---------------------------------------------------------------------------

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
        "- ارسال لینک: یک لینک ویدیویی برای من بفرست؛ ربات ویدیو را دانلود، صوت را استخراج، متن را تشخیص و ترجمه می‌کنه.\n\n"
        "- بعد از پردازش از شما می‌پرسد خروجی چطور باشد:\n"
        "  📹 کلیپ با زیرنویس فارسی\n"
        "  📝 فقط متن دوزبانه (سریع‌تر، بدون آپلود ویدیو)\n\n"
        "- فرمان‌ها: /start برای خوش‌آمدگویی، /help برای نمایش این راهنما.\n\n"
        "- محدودیت‌ها: کلیپ های بالای یک گیگ رو نمیتونیم پردازش کنیم فعلا، تعداد پردازش همزمان کلیپ ها محدودع، پس اگه طول میکشه یکم صبر کن.\n\n"
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

    status_msg = await safe_telegram_call(
        reply_text_factory(update.message, "⏳ لینک شما به صف پردازش اضافه شد. در حال دانلود، تشخیص گفتار و ترجمه...")
    )

    try:
        async with PROCESS_SEMAPHORE:
            prepared = await prepare_video(url, user_info)

        # Ask user what kind of output they want
        job_id = str(uuid.uuid4())
        store_pending_job(context.application.bot_data, job_id, {
            "url": url,
            "user_info": user_info,
            "video_path": prepared["video_path"],
            "srt_path": prepared["srt_path"],
            "segments": prepared["segments"],
            "original_message_id": update.message.message_id,
            "chat_id": update.effective_chat.id,
        })

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📹 کلیپ با زیرنویس", callback_data=f"clip:{job_id}"),
                InlineKeyboardButton("📝 فقط متن دوزبانه", callback_data=f"text:{job_id}"),
            ]
        ])

        await safe_telegram_call(edit_text_factory(
            status_msg,
            "✅ پردازش کامل شد! خروجی مورد نظرتان را انتخاب کنید:"
        ))
        await safe_telegram_call(reply_markup_factory(
            update.message,
            "🎯 چه نوع خروجی می‌خواهید؟",
            keyboard,
        ))

    except Exception as e:
        log_action("Telegram", user_info, "FAILED", url)
        logger.exception("Processing failed for %s", url)
        await safe_telegram_call(
            reply_text_factory(update.message, f"❌ خطایی در طول فرآیند پردازش رخ داد: {str(e)}")
        )


# ---------------------------------------------------------------------------
# Callback handler for output-type selection
# ---------------------------------------------------------------------------

async def handle_output_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # acknowledge the button tap immediately

    data = query.data or ""
    if ":" not in data:
        logger.warning("Unexpected callback data: %s", data)
        return

    choice, job_id = data.split(":", 1)
    job = pop_pending_job(context.application.bot_data, job_id)

    if job is None:
        logger.warning("Callback for unknown/expired job_id=%s choice=%s", job_id, choice)
        await query.edit_message_text("⚠️ این درخواست منقضی شده یا قبلاً پردازش شده است.")
        return

    url = job["url"]
    user_info = job["user_info"]
    video_path = job["video_path"]
    srt_path = job["srt_path"]
    segments = job["segments"]

    logger.info("User %s chose output type '%s' for job %s", user_info, choice, job_id)

    if choice == "clip":
        # --- Clip with subtitles ---
        await query.edit_message_text("⏳ در حال میکس زیرنویس با ویدیو، لطفاً صبر کنید...")
        try:
            final_mkv = await mux_clip(video_path, srt_path, url, user_info)
            await query.edit_message_text("📤 آماده شد! در حال ارسال فایل ویدیو (MKV)...")

            # Send the video file; reply to the original message for context
            async def send_clip():
                with open(final_mkv, 'rb') as f:
                    return await context.bot.send_document(
                        chat_id=job["chat_id"],
                        document=f,
                        caption="🎬 کلیپ با زیرنویس فارسی",
                    )
            await safe_telegram_call(send_clip)
            await query.edit_message_text("✅ ویدیو با زیرنویس فارسی ارسال شد.")
            log_action("Telegram", user_info, "DELIVERED (CLIP)", url, final_mkv)

        except Exception as e:
            logger.exception("Muxing/delivery failed for job %s", job_id)
            # Cleanup on failure
            cleanup_prepare_files(video_path, srt_path)
            await query.edit_message_text(f"❌ خطا در ساخت کلیپ: {str(e)}")

    elif choice == "text":
        # --- Dual-language text only, no mux ---
        log_action("Telegram", user_info, "DELIVERING TEXT", url)
        cleanup_prepare_files(video_path, srt_path)

        chunks = build_dual_text(segments)
        if not chunks:
            await query.edit_message_text("⚠️ متنی برای ارسال وجود ندارد.")
            return

        await query.edit_message_text(f"📝 در حال ارسال متن دوزبانه ({len(chunks)} بخش)...")

        for idx, chunk in enumerate(chunks, start=1):
            async def send_chunk(c=chunk):
                return await context.bot.send_message(
                    chat_id=job["chat_id"],
                    text=c,
                )
            await safe_telegram_call(send_chunk)
            # Small delay to avoid hitting rate limits when many chunks
            if idx < len(chunks):
                await asyncio.sleep(0.5)

        await query.edit_message_text(f"✅ متن دوزبانه ارسال شد ({len(chunks)} بخش).")
        log_action("Telegram", user_info, "DELIVERED (TEXT)", url)

    else:
        logger.warning("Unknown choice '%s' for job %s", choice, job_id)
        cleanup_prepare_files(video_path, srt_path)
        await query.edit_message_text("⚠️ گزینه نامعتبر.")


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
    app.add_handler(CallbackQueryHandler(handle_output_choice))
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