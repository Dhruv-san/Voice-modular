import os
import torch
import argparse
import gradio as gr
import langid
import requests
import zipfile
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args, unknown = parser.parse_known_args()

# Constants and Paths
# On Vercel, we should use /tmp for large downloads and processed files
IS_VERCEL = "VERCEL" in os.environ
BASE_DIR = "/tmp/star_ai" if IS_VERCEL else os.getcwd()
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

en_ckpt_base = os.path.join(CHECKPOINT_DIR, 'base_speakers/EN')
zh_ckpt_base = os.path.join(CHECKPOINT_DIR, 'base_speakers/ZH')
ckpt_converter = os.path.join(CHECKPOINT_DIR, 'converter')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global variables for models
en_base_speaker_tts = None
zh_base_speaker_tts = None
tone_color_converter = None
en_source_default_se = None
en_source_style_se = None
zh_source_se = None

def download_checkpoints():
    if os.path.exists(CHECKPOINT_DIR):
        return True

    print("Downloading checkpoints...")
    url = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip"
    r = requests.get(url, stream=True)
    zip_path = os.path.join(BASE_DIR, "checkpoints.zip")
    os.makedirs(BASE_DIR, exist_ok=True)

    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)

    os.remove(zip_path)
    return True

def load_models():
    global en_base_speaker_tts, zh_base_speaker_tts, tone_color_converter
    global en_source_default_se, en_source_style_se, zh_source_se

    try:
        download_checkpoints()

        en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
        en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
        zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
        zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth', map_location=device)
        en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth', map_location=device)
        zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth', map_location=device)
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Lazy loading for Vercel start-up performance
models_loaded = False

supported_languages = ['zh', 'en']

def predict(prompt, style, audio_file_pth, agree):
    global models_loaded
    if not models_loaded:
        if load_models():
            models_loaded = True
        else:
            return "[ERROR] Models could not be loaded. Check logs.", None, None

    text_hint = ''
    if not agree:
        text_hint += '[ERROR] Please accept the Terms & Condition!\n'
        gr.Warning("Please accept the Terms & Condition!")
        return text_hint, None, None

    if not audio_file_pth:
        text_hint += '[ERROR] Please upload a reference audio!\n'
        gr.Warning("Please upload a reference audio!")
        return text_hint, None, None

    language_predicted = langid.classify(prompt)[0].strip()

    if language_predicted not in supported_languages:
        error_msg = f"The detected language {language_predicted} is not supported. Supported: {supported_languages}"
        text_hint += f"[ERROR] {error_msg}\n"
        gr.Warning(error_msg)
        return text_hint, None, None

    if language_predicted == "zh":
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        language = 'Chinese'
        if style not in ['default']:
            error_msg = f"The style {style} is not supported for Chinese."
            text_hint += f"[ERROR] {error_msg}\n"
            gr.Warning(error_msg)
            return text_hint, None, None
    else:
        tts_model = en_base_speaker_tts
        source_se = en_source_default_se if style == 'default' else en_source_style_se
        language = 'English'
        if style not in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']:
            error_msg = f"The style {style} is not supported for English."
            text_hint += f"[ERROR] {error_msg}\n"
            gr.Warning(error_msg)
            return text_hint, None, None

    if len(prompt) < 2 or len(prompt) > 200:
        error_msg = "Prompt length must be between 2 and 200 characters."
        text_hint += f"[ERROR] {error_msg}\n"
        gr.Warning(error_msg)
        return text_hint, None, None

    try:
        # Use our /tmp processed dir
        target_se, _ = se_extractor.get_se(audio_file_pth, tone_color_converter, target_dir=PROCESSED_DIR, vad=True)
    except Exception as e:
        error_msg = f"Get target tone color error: {e}"
        text_hint += f"[ERROR] {error_msg}\n"
        gr.Warning(error_msg)
        return text_hint, None, None

    src_path = os.path.join(OUTPUT_DIR, 'tmp.wav')
    tts_model.tts(prompt, src_path, speaker=style, language=language)

    save_path = os.path.join(OUTPUT_DIR, 'output.wav')
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message="@StarAI"
    )

    text_hint += "Success! Audio generated.\n"
    return text_hint, save_path, audio_file_pth

# --- UI Design ---

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

body { background-color: #ffffff; font-family: 'Inter', sans-serif; color: #000000; }
.gradio-container { max-width: 800px !important; margin: auto; padding: 20px; border: none !important; box-shadow: none !important; }
#star-header { text-align: center; padding: 60px 20px; background: #ffffff; color: #000080; border-radius: 0; margin-bottom: 20px; border-bottom: 1px solid #eeeeee; animation: fadeInDown 1.5s cubic-bezier(0.22, 1, 0.36, 1); }
#star-header h1 { font-weight: 600; letter-spacing: 12px; margin: 0; font-size: 4em; text-transform: uppercase; color: #000000; }
#star-header p { font-weight: 300; opacity: 0.6; font-size: 1em; margin-top: 15px; letter-spacing: 2px; }
@keyframes fadeInDown { from { opacity: 0; transform: translateY(-40px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.gr-button-primary { background: #000080 !important; color: #ffffff !important; border: none !important; border-radius: 4px !important; padding: 14px 28px !important; font-weight: 400 !important; letter-spacing: 1px !important; transition: all 0.3s ease !important; text-transform: uppercase !important; font-size: 0.9em !important; }
.gr-button-primary:hover { background: #000000 !important; transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; }
.gr-form { border: none !important; background: #ffffff !important; }
.gr-input, .gr-select, .gr-file { border-radius: 0px !important; border: 1px solid #e0e0e0 !important; background: #ffffff !important; }
#motion-box { width: 100%; height: 120px; display: flex; justify-content: center; align-items: center; margin-bottom: 40px; overflow: hidden; }
.star { fill: #000080; animation: floating 4s infinite ease-in-out; }
@keyframes floating { 0% { transform: translateY(0px) rotate(0deg); opacity: 0.2; } 50% { transform: translateY(-20px) rotate(180deg); opacity: 0.8; } 100% { transform: translateY(0px) rotate(360deg); opacity: 0.2; } }
.tab-nav { border-bottom: 1px solid #eeeeee !important; justify-content: center !important; }
.tab-nav button { border: none !important; font-weight: 300 !important; color: #999999 !important; padding: 15px 30px !important; letter-spacing: 1px !important; }
.tab-nav button.selected { color: #000080 !important; background: transparent !important; border-bottom: 2px solid #000080 !important; }
footer { display: none !important; }
"""

motion_graphics_html = """
<div id="motion-box">
    <svg width="400" height="100" viewBox="0 0 400 100">
        <path class="star" d="M200 30 L204 42 L216 42 L206 50 L210 62 L200 54 L190 62 L194 50 L184 42 L196 42 Z" />
        <circle class="star" cx="150" cy="50" r="1" style="animation-delay: 0.2s;"/>
        <circle class="star" cx="250" cy="50" r="1" style="animation-delay: 0.8s;"/>
    </svg>
</div>
"""

with gr.Blocks(css=css, title="STAR AI") as demo:
    gr.HTML("""<div id="star-header"><h1>STAR AI</h1><p>V O I C E . C L O N I N G . R E D E F I N E D</p></div>""")
    gr.HTML(motion_graphics_html)

    with gr.Tabs():
        with gr.TabItem("EXPERIENCE"):
            gr.Markdown("""<div style="text-align: center; padding: 40px 0;"><h2 style="font-weight: 300; letter-spacing: 4px;">THE ESSENCE OF VOICE</h2><p style="color: #666; max-width: 500px; margin: auto; line-height: 1.8;">Experience the absolute purity of tone replication.</p></div>""")

        with gr.TabItem("CREATE"):
            with gr.Column():
                input_text = gr.Textbox(label="TEXT PROMPT", placeholder="Sequence to synthesize...", lines=2)
                style = gr.Dropdown(label="MODALITY", choices=['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly'], value="default")
                ref_audio = gr.Audio(label="SOURCE REFERENCE", type="filepath")
                tos = gr.Checkbox(label="ACCEPT CC-BY-NC-4.0 TERMS", value=False)
                generate_btn = gr.Button("INITIALIZE SYNTHESIS", variant="primary")
                audio_output = gr.Audio(label="SYNTHESIZED OUTPUT", interactive=False, autoplay=True)
                info_output = gr.Textbox(label="SYSTEM LOG", interactive=False)
                ref_audio_used = gr.Audio(label="REFERENCE ARCHIVE", interactive=False)

                gr.Examples(
                    examples=[
                        ['今天天气真好，我们一起出去吃饭吧。', 'default', 'resources/demo_speaker1.mp3', True],
                        ['Distilled voice quality with zero-shot adaptation.', 'whispering', 'resources/demo_speaker2.mp3', True],
                    ],
                    inputs=[input_text, style, ref_audio, tos],
                    outputs=[info_output, audio_output, ref_audio_used],
                    fn=predict, cache_examples=False
                )

    generate_btn.click(fn=predict, inputs=[input_text, style, ref_audio, tos], outputs=[info_output, audio_output, ref_audio_used])

if __name__ == "__main__":
    demo.queue().launch(share=args.share)
