import os
import sys
import zipfile
import urllib.request
import subprocess

# Set up paths
sys.path.append(os.getcwd())

def download_checkpoints():
    # We'll skip the auto-download if the link is broken,
    # and let the user handle it or provide a better link later.
    # But we'll keep the logic for when they do provide it.
    checkpoints_dir = 'checkpoints'
    if not os.path.exists(os.path.join(checkpoints_dir, 'base_speakers')):
        print("Checkpoints directory 'checkpoints/base_speakers' not found.")
        print("Please ensure checkpoints are downloaded and placed in the 'checkpoints' folder.")
        # Attempting a known working link if possible, or just informing the user.
        # The link in README might be outdated.

def main():
    # Download checkpoints if missing
    download_checkpoints()

    # Set environment variables for Gradio
    os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
    os.environ['GRADIO_SERVER_PORT'] = os.environ.get('PORT', '10000')

    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)

    print("Launching OpenVoice app...")
    try:
        from openvoice import openvoice_app
    except Exception as e:
        print(f"Error launching app: {e}")
        # import traceback
        # traceback.print_exc()
        # sys.exit(1)

if __name__ == "__main__":
    main()
