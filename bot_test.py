import os
import sys

# --- پاکسازی متغیرهای مخفی ترمینال اوبونتو قبل از لود شدن کتابخانه‌ها ---
PROXY_URL = "http://127.0.0.1:2080"
for key in list(os.environ.keys()):
    if "proxy" in key.lower():
        del os.environ[key]

# تنظیم دستی و اجباری پروکسی برای بقیه ابزارها (yt-dlp و deep-translator)
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL

import json
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
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

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    print("💥 Error: BOT_TOKEN not found in .env file!", flush=True)
    sys.exit(1)

from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
import yt_dlp

MODEL_PATH = "./whisper_model"
CACHE_DIR = "./cache"
ARCHIVE_DIR = "./archive"
HISTORY_FILE = "./history.json"
LOG_FILE = "./service.log"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

def log_action(source, user_info, action, url, file_path="N/A"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [SOURCE: {source}] [USER: {user_info}] [ACTION: {action}] [URL: {url}] [FILE: {file_path}]"
    print(log_line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

def get_archived_file(url):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
            return history.get(url)
    return None

def save_to_history(url, file_path):
    history = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass
    history[url] = file_path
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

async def process_video(url, user_info):
    base_name = f"video_{int(datetime.now().timestamp())}"
    video_path = os.path.join(CACHE_DIR, f"{base_name}.mp4")
    audio_path = os.path.join(CACHE_DIR, f"{base_name}.wav")
    srt_path = os.path.join(CACHE_DIR, f"{base_name}.srt")
    final_mkv = os.path.join(ARCHIVE_DIR, f"{base_name}.mkv")

    # ۱. دانلود ویدیو با بهترین کیفیت ممکن با فرمت MP4
    log_action("Telegram", user_info, "DOWNLOADING", url)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': video_path,
        'proxy': PROXY_URL,
        'quiet': True
    }
    
    # اجرای دانلود ویدیو
    await asyncio.to_thread(lambda: yt_dlp.YoutubeDL(ydl_opts).download([url]))

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"ویدیو دانلود شد اما در مسیر مشخص شده یافت نشد: {video_path}")

    # ۲. استخراج صدای ۱۶کیلوهرتز مونو با استفاده مستقیم از ffmpeg (بسیار امن و سریع)
    log_action("Telegram", user_info, "EXTRACTING AUDIO", url)
    audio_cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}"'
    audio_proc = await asyncio.create_subprocess_shell(audio_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await audio_proc.communicate()

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"خطا در استخراج فایل صوتی توسط ffmpeg: {audio_path}")

    # ۳. تشخیص گفتار با استفاده از مدل لوکال Whisper
    log_action("Telegram", user_info, "TRANSCRIBING", url)
    model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4, download_root=MODEL_PATH)
    segments, _ = await asyncio.to_thread(model.transcribe, audio_path, beam_size=5)
    segments = list(segments)

    # ۴. ترجمه بخش‌ها به فارسی آنلاین
    log_action("Telegram", user_info, "TRANSLATING", url)
    translator = GoogleTranslator(source='auto', target='fa', proxies={"http": PROXY_URL, "https": PROXY_URL})
    
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments, start=1):
            translated_text = await asyncio.to_thread(translator.translate, segment.text)
            start_time = format_srt_time(segment.start)
            end_time = format_srt_time(segment.end)
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{translated_text}\n\n")

    # ۵. میکس نهایی ویدیو و زیرنویس متنی به صورت سافت‌کد بدون رندر مجدد ویدیو
    log_action("Telegram", user_info, "MUXING", url)
    mux_cmd = f'ffmpeg -y -i "{video_path}" -i "{srt_path}" -c copy -c:s srt "{final_mkv}"'
    
    proc = await asyncio.create_subprocess_shell(mux_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.communicate()

    # ۶. پاکسازی فایل‌های موقت کش
    for path in [video_path, audio_path, srt_path]:
        if os.path.exists(path):
            os.remove(path)

    save_to_history(url, final_mkv)
    log_action("Telegram", user_info, "COMPLETED", url, final_mkv)
    return final_mkv

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_info = f"@{user.username} (ID: {user.id})" if user.username else f"ID: {user.id}"
    log_action("Telegram", user_info, "COMMAND: /start", "N/A")
    
    await update.message.reply_text(
        "👋 سلام! به ربات مترجم ویدیو خوش آمدید.\n\n"
        "🔗 کافیست لینک ویدیو یا کلیپ خود را برای من بفرستید تا آن را با زیرنویس فارسی (Softcode MKV) تحویلتان دهم."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"📥 Received a raw text string: {update.message.text}", flush=True)
    
    url = update.message.text.strip()
    if not url.startswith(("http://", "https://")):
        print(f"⚠️ Ignored message because it doesn't look like a valid link.", flush=True)
        return

    user = update.message.from_user
    user_info = f"@{user.username} (ID: {user.id})" if user.username else f"ID: {user.id}"
    
    log_action("Telegram", user_info, "RECEIVED", url)

    cached_file = get_archived_file(url)
    if cached_file and os.path.exists(cached_file):
        log_action("Telegram", user_info, "CACHE_HIT", url, cached_file)
        await update.message.reply_text("✨ این لینک قبلاً پردازش شده است! در حال ارسال فایل اصلی آرشیو...")
        await update.message.reply_document(document=open(cached_file, 'rb'))
        return

    status_msg = await update.message.reply_text("⏳ لینک شما به صف پردازش اضافه شد. در حال دانلود و آماده‌سازی زیرنویس فارسی...")

    try:
        output_mkv = await process_video(url, user_info)
        await status_msg.edit_text("📤 فرآیند ترجمه به پایان رسید! در حال ارسال فایل ویدیو (Uncompressed MKV)...")
        await update.message.reply_document(document=open(output_mkv, 'rb'))
    except Exception as e:
        log_action("Telegram", user_info, "FAILED", url)
        print(f"💥 Critical Failure Details: {str(e)}", flush=True)
        await update.message.reply_text(f"❌ خطایی در طول فرآیند پردازش رخ داد: {str(e)}")

def main():
    print("🚀 Local pipeline bot testing script initialized...", flush=True)
    print(f"🌐 Explicitly forcing HTTP Proxy network bridge: {PROXY_URL}", flush=True)
    
    try:
        app = (
            Application.builder()
            .token(BOT_TOKEN)
            .proxy(PROXY_URL)
            .get_updates_proxy(PROXY_URL)
            .build()
        )
        
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        print("🔋 Listening for message links incoming via Telegram...", flush=True)
        app.run_polling()
    except Exception as e:
        print(f"💥 Failed to boot polling client: {str(e)}", flush=True)

if __name__ == "__main__":
    main()