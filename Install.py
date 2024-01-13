import tkinter as tk
from tkinter import messagebox
import requests
import zipfile
import subprocess
import sys
import os

def install_audiocraft():
    try:
        # Ensure PyTorch is installed first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.1.0"])
        
        # Install setuptools and wheel
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools", "wheel"])

        # Fetching the latest release from GitHub
        audiocraft_repo = "facebookresearch/audiocraft"
        api_url = f"https://api.github.com/repos/{audiocraft_repo}/releases/latest"
        response = requests.get(api_url).json()
        download_url = response['assets'][0]['browser_download_url']
        
        # Downloading the release
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open("audiocraft.zip", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)

        # Extracting the release
        with zipfile.ZipFile("audiocraft.zip", "r") as zip_ref:
            zip_ref.extractall("audiocraft_installation")

        # Install AudioCraft and its Python dependencies
        requirements_path = os.path.join("audiocraft_installation", "requirements.txt")
        if os.path.exists(requirements_path):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        else:
            # Install AudioCraft directly (if requirements.txt is not found)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "audiocraft"])

        messagebox.showinfo("Install", "AudioCraft installation completed.")

    except Exception as e:
        messagebox.showerror("Error", f"Installation failed: {e}")

    finally:
        # Clean up the downloaded zip file if it exists
        if os.path.exists("audiocraft.zip"):
            os.remove("audiocraft.zip")

def run_audiocraft():
    

def exit_app():
    

root = tk.Tk()

