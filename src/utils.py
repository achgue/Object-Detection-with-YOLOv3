import os
import wget


# Download function for local execution
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        wget.download(url, filename)
    else:
        print(f"{filename} already exists.")