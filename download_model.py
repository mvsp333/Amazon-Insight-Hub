import gdown

def download_model():
    # Replace this with your actual Google Drive file ID
    file_id = "1UCAcMhzZaVfsVmxF8ksPCmL5yobHiXrg"
    
    # Construct the download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Local filename to save
    output = "randon_forest_regressor.pkl"
    
    print("Downloading model from Google Drive...")
    gdown.download(url, output, quiet=False)
    print(f"Model downloaded and saved as '{output}'")

if __name__ == "__main__":
    download_model()
