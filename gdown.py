import os
import gdown

def download_file(url, output):
    if not os.path.exists(output):
        print(f"Downloading {output}...")
        gdown.download(url, output, quiet=False)

# Define your file URLs and output paths
files = {
    "model.safetensors": "https://drive.google.com/uc?id=MODEL_FILE_ID",
    "optimizer.pt": "https://drive.google.com/uc?id=OPTIMIZER_FILE_ID",
}

# Create the directory if it doesn't exist
os.makedirs("gpt2-finetuned/checkpoint-3", exist_ok=True)

# Download each file
for filename, url in files.items():
    output_path = os.path.join("gpt2-finetuned/checkpoint-3", filename)
    download_file(url, output_path)