import os
from faster_whisper import WhisperModel

def download_local_model():
    model_size = "base"  # You can change this to "tiny" or "small"
    output_dir = "./whisper_model"
    
    print(f"📥 Downloading Whisper '{model_size}' model to '{output_dir}'...")
    
    # This downloads and caches the model files locally into the specified directory
    model = WhisperModel(
        model_size, 
        device="cpu", 
        compute_type="int8", 
        download_root=output_dir
    )
    
    print("✅ Model successfully downloaded and verified locally!")

if __name__ == "__main__":
    download_local_model()