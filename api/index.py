import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from star_ai_app import demo

# Vercel's @vercel/python builder expects an 'app' variable if it's a WSGI/ASGI app
# Gradio apps can be converted to ASGI apps using demo.app
app = demo.app
