import os
from urllib.parse import urlparse
from faster_whisper import WhisperModel


def download_local_model():
    model_size = "base"  # You can change this to "tiny" or "small"
    output_dir = os.getenv("MODEL_PATH", "./whisper_model")

    # Respect only HTTP/HTTPS proxies for the downloader. Ignore socks proxies (they may be unsupported).
    proxy_url = os.getenv("PROXY_URL", "")
    if proxy_url:
        parsed = urlparse(proxy_url)
        if parsed.scheme in ("http", "https"):
            os.environ.setdefault("HTTP_PROXY", proxy_url)
            os.environ.setdefault("HTTPS_PROXY", proxy_url)
            print(f"Using HTTP proxy for download: {proxy_url}")
        else:
            print(f"Warning: PROXY_URL has unsupported scheme '{parsed.scheme}'. Ignoring proxy for download.")

    print(f"📥 Downloading Whisper '{model_size}' model to '{output_dir}'...")

    # If model directory already has files, skip download
    if os.path.exists(output_dir) and any(os.scandir(output_dir)):
        print(f"✅ Local model already present in '{output_dir}', skipping download.")
        return

    try:
        # This downloads and caches the model files locally into the specified directory
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=output_dir,
        )
        print("✅ Model successfully downloaded and verified locally!")
    except Exception as e:
        print(f"Failed to download or initialize WhisperModel: {e}")
        raise


if __name__ == "__main__":
    download_local_model()