import os
import gdown

def download_model():
    file_id = "18fAQ62gvI91JKEmwtIORtNsa5--b0qD6"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "brainNet\Plane_detector_model.pth" 

    if not os.path.exists(output_path):
        print("🔽 Downloading model...")
        gdown.download(url, output_path, quiet=False)

    if os.path.exists(output_path):
        print("✅ Model downloaded.")
        print("📦 Size:", os.path.getsize(output_path) / (1024 * 1024), "MB")
    else:
        print("❌ Download failed.")

download_model()
