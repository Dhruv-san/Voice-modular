import os
import torch
import argparse
import gradio as gr
import langid
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Constants and Paths
en_ckpt_base = 'checkpoints/base_speakers/EN'
zh_ckpt_base = 'checkpoints/base_speakers/ZH'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load Models
try:
    en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
    en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
    zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
    zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    # Load speaker embeddings
    en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
    en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)
    zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth').to(device)
except Exception as e:
    print(f"Warning: Models could not be loaded. Please ensure checkpoints are in place. Error: {e}")

supported_languages = ['zh', 'en']

def predict(prompt, style, audio_file_pth, agree):
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
    print(f"Detected language: {language_predicted}")

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

    if len(prompt) < 2:
        error_msg = "Please give a longer prompt text."
        text_hint += f"[ERROR] {error_msg}\n"
        gr.Warning(error_msg)
        return text_hint, None, None

    if len(prompt) > 200:
        error_msg = "Text length limited to 200 characters."
        text_hint += f"[ERROR] {error_msg}\n"
        gr.Warning(error_msg)
        return text_hint, None, None

    try:
        target_se, _ = se_extractor.get_se(audio_file_pth, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        error_msg = f"Get target tone color error: {e}"
        text_hint += f"[ERROR] {error_msg}\n"
        gr.Warning(error_msg)
        return text_hint, None, None

    src_path = f'{output_dir}/tmp.wav'
    tts_model.tts(prompt, src_path, speaker=style, language=language)

    save_path = f'{output_dir}/output.wav'
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

body {
    background-color: #ffffff;
    font-family: 'Inter', sans-serif;
    color: #000000;
}

.gradio-container {
    max-width: 800px !important;
    margin: auto;
    padding: 20px;
    border: none !important;
    box-shadow: none !important;
}

#star-header {
    text-align: center;
    padding: 80px 20px;
    background: #ffffff;
    color: #000080;
    border-radius: 0;
    margin-bottom: 20px;
    border-bottom: 1px solid #eeeeee;
    animation: fadeInDown 1.5s cubic-bezier(0.22, 1, 0.36, 1);
}

#star-header h1 {
    font-weight: 600;
    letter-spacing: 12px;
    margin: 0;
    font-size: 4em;
    text-transform: uppercase;
    color: #000000;
}

#star-header p {
    font-weight: 300;
    opacity: 0.6;
    font-size: 1em;
    margin-top: 15px;
    letter-spacing: 2px;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-40px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.tabs-container {
    animation: fadeIn 1.2s ease-in;
}

/* Minimalistic Button styling */
.gr-button-primary {
    background: #000080 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 14px 28px !important;
    font-weight: 400 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    font-size: 0.9em !important;
}

.gr-button-primary:hover {
    background: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}

/* Form and Box styling */
.gr-form {
    border: none !important;
    background: #ffffff !important;
}

.gr-input, .gr-select, .gr-file {
    border-radius: 0px !important;
    border: 1px solid #e0e0e0 !important;
    background: #ffffff !important;
}

.gr-input:focus, .gr-select:focus {
    border-color: #000080 !important;
    box-shadow: none !important;
}

#motion-box {
    width: 100%;
    height: 120px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 40px;
    overflow: hidden;
}

.star {
    fill: #000080;
    animation: floating 4s infinite ease-in-out;
}

@keyframes floating {
    0% { transform: translateY(0px) rotate(0deg); opacity: 0.2; }
    50% { transform: translateY(-20px) rotate(180deg); opacity: 0.8; }
    100% { transform: translateY(0px) rotate(360deg); opacity: 0.2; }
}

.tab-nav {
    border-bottom: 1px solid #eeeeee !important;
    justify-content: center !important;
}

.tab-nav button {
    border: none !important;
    font-weight: 300 !important;
    color: #999999 !important;
    padding: 15px 30px !important;
    letter-spacing: 1px !important;
}

.tab-nav button.selected {
    color: #000080 !important;
    background: transparent !important;
    border-bottom: 2px solid #000080 !important;
}

footer { display: none !important; }
"""

motion_graphics_html = """
<div id="motion-box">
    <svg width="400" height="100" viewBox="0 0 400 100">
        <!-- Central Star -->
        <path class="star" d="M200 30 L204 42 L216 42 L206 50 L210 62 L200 54 L190 62 L194 50 L184 42 L196 42 Z" />
        <!-- Surrounding accents -->
        <circle class="star" cx="150" cy="50" r="1" style="animation-delay: 0.2s; animation-duration: 5s;"/>
        <circle class="star" cx="250" cy="50" r="1" style="animation-delay: 0.8s; animation-duration: 6s;"/>
        <circle class="star" cx="100" cy="30" r="0.5" style="animation-delay: 1.5s; animation-duration: 7s;"/>
        <circle class="star" cx="300" cy="70" r="0.5" style="animation-delay: 0.4s; animation-duration: 4s;"/>
    </svg>
</div>
"""

with gr.Blocks(css=css, title="STAR AI") as demo:
    # Header Section
    gr.HTML("""
        <div id="star-header">
            <h1>STAR AI</h1>
            <p>V O I C E . C L O N I N G . R E D E F I N E D</p>
        </div>
    """)

    gr.HTML(motion_graphics_html)

    with gr.Tabs(elem_id="tabs"):
        # Tab 1: Experience
        with gr.TabItem("EXPERIENCE"):
            with gr.Column():
                gr.Markdown("""
                <div style="text-align: center; padding: 40px 0;">
                    <h2 style="font-weight: 300; letter-spacing: 4px;">THE ESSENCE OF VOICE</h2>
                    <p style="color: #666; max-width: 500px; margin: auto; line-height: 1.8;">
                        Experience the absolute purity of tone replication.
                        Star AI distills the human voice into its core components,
                        allowing for seamless reconstruction in any context.
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        <div style="text-align: center; border-top: 1px solid #eee; padding: 20px;">
                            <h4 style="letter-spacing: 2px;">01. CLONE</h4>
                            <p style="font-size: 0.85em; color: #888;">Instant extraction of tone color embeddings.</p>
                        </div>
                        """)
                    with gr.Column():
                        gr.Markdown("""
                        <div style="text-align: center; border-top: 1px solid #eee; padding: 20px;">
                            <h4 style="letter-spacing: 2px;">02. ADAPT</h4>
                            <p style="font-size: 0.85em; color: #888;">Fluid emotional style conversion.</p>
                        </div>
                        """)
                    with gr.Column():
                        gr.Markdown("""
                        <div style="text-align: center; border-top: 1px solid #eee; padding: 20px;">
                            <h4 style="letter-spacing: 2px;">03. SPEAK</h4>
                            <p style="font-size: 0.85em; color: #888;">Multi-lingual synthesis with natural resonance.</p>
                        </div>
                        """)

        # Tab 2: Create
        with gr.TabItem("CREATE"):
            with gr.Column():
                with gr.Row():
                    input_text = gr.Textbox(
                        label="TEXT PROMPT",
                        placeholder="Sequence to synthesize...",
                        lines=2
                    )

                with gr.Row():
                    style = gr.Dropdown(
                        label="MODALITY",
                        choices=['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly'],
                        value="default"
                    )

                with gr.Row():
                    ref_audio = gr.Audio(
                        label="SOURCE REFERENCE",
                        type="filepath"
                    )

                with gr.Row():
                    tos = gr.Checkbox(
                        label="ACCEPT CC-BY-NC-4.0 TERMS",
                        value=False
                    )

                generate_btn = gr.Button("INITIALIZE SYNTHESIS", variant="primary")

                gr.Markdown("<br>")

                with gr.Row():
                    audio_output = gr.Audio(label="SYNTHESIZED OUTPUT", interactive=False, autoplay=True)

                with gr.Row():
                    info_output = gr.Textbox(label="SYSTEM LOG", interactive=False)
                    ref_audio_used = gr.Audio(label="REFERENCE ARCHIVE", interactive=False)

                gr.Examples(
                    examples=[
                        ['今天天气真好，我们一起出去吃饭吧。', 'default', 'resources/demo_speaker1.mp3', True],
                        ['Distilled voice quality with zero-shot adaptation.', 'whispering', 'resources/demo_speaker2.mp3', True],
                        ['A silent ocean under the midnight stars.', 'sad', 'resources/demo_speaker0.mp3', True]
                    ],
                    inputs=[input_text, style, ref_audio, tos],
                    outputs=[info_output, audio_output, ref_audio_used],
                    fn=predict,
                    cache_examples=False
                )

        # Tab 3: Insight
        with gr.TabItem("INSIGHT"):
            with gr.Column():
                gr.Markdown("""
                <div style="padding: 20px; line-height: 2;">
                    <h3 style="font-weight: 400; letter-spacing: 2px;">TECHNOLOGY</h3>
                    <p style="color: #666;">Star AI is built upon the OpenVoice architecture, utilizing high-performance base speakers and a sophisticated tone color converter. It achieves granular control over style parameters including rhythm, pauses, and intonation.</p>

                    <h3 style="font-weight: 400; letter-spacing: 2px; margin-top: 30px;">ETHOS</h3>
                    <p style="color: #666;">We believe in the harmony of technology and ethics. Star AI is designed for creative exploration, research, and accessible communication. We advocate for the respectful use of vocal identity.</p>
                </div>
                """)

    # Event Handlers
    generate_btn.click(
        fn=predict,
        inputs=[input_text, style, ref_audio, tos],
        outputs=[info_output, audio_output, ref_audio_used]
    )

if __name__ == "__main__":
    demo.queue().launch(share=args.share)
