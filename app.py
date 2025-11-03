import os
import torch

# CRITICAL: Redirect cache to temporary storage to avoid hitting storage limits
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['TMPDIR'] = '/tmp'
torch.hub.set_dir('/tmp/torch_hub')

import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load models with priority to YOLOv8
# Try to load YOLOv8 model first, fall back to YOLOv11 if not available
model = None
model_name = ""

if os.path.exists("best.pt"):
    model = YOLO("best.pt")
    model_name = "YOLOv8 (best.pt)"
    print("✓ Loaded YOLOv8 model (best.pt)")
elif os.path.exists("yolov11nbest.pt"):
    model = YOLO("yolov11nbest.pt")
    model_name = "YOLOv11 (yolov11nbest.pt)"
    print("✓ Loaded YOLOv11 model (yolov11nbest.pt)")
else:
    raise FileNotFoundError("No model file found. Please ensure 'best.pt' or 'yolov11nbest.pt' exists.")

# Define the prediction function with progress updates
def predict(image, progress=gr.Progress()):
    progress(0, desc="Starting detection...")

    progress(0.3, desc="Running AI model (this may take 20-40 seconds on CPU)...")
    results = model(image)  # Run YOLO model on the uploaded image

    progress(0.8, desc="Drawing bounding boxes...")
    results_img = results[0].plot()  # Get image with bounding boxes

    progress(1.0, desc="Done!")
    return Image.fromarray(results_img)

# Get example images from the images folder
def get_example_images():
    examples = []
    image_folder = "images"
    if os.path.exists(image_folder):
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                examples.append(os.path.join(image_folder, filename))
    return examples

# Custom CSS for better visual appeal
custom_css = """
.progress-bar-wrap {
    border-radius: 8px !important;
}
.progress-bar {
    background: linear-gradient(90deg, #4CAF50, #2196F3) !important;
}
"""

# Create Gradio interface with better UX
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📤 Upload Image"),
    outputs=gr.Image(type="pil", label="🎯 Detection Result"),
    title="🪖 Helmet Detection with YOLO",
    description=f"""
    **Currently using: {model_name}**

    Upload an image to detect helmets and safety equipment. The AI will identify:
    - ✅ People wearing helmets (accept-Helmet-)
    - ❌ People not wearing helmets (non-Helmet-)

    ⏱️ **Please be patient**: Detection takes 20-40 seconds on CPU. Watch the progress bar below!
    """,
    article="""
    ### 📊 About This Model
    This app uses YOLOv8/YOLOv11 for real-time helmet detection in construction and workplace safety scenarios.

    **Tips for best results:**
    - Use clear, well-lit images
    - Ensure people are visible in the frame
    - Works best with construction site or workplace photos

    **Performance:** Running on CPU (free tier), so inference takes ~30 seconds per image.

    ### ⭐ Advanced Version Available!
    Check out the **[Helmet + License Plate Detection](https://huggingface.co/spaces/Abs6187/Helmet-License-Plate-Detection)** -
    an upgraded version that detects BOTH helmets AND license plates!

    ### 🚀 While you wait...
    - Try the example images below
    - Read about YOLO object detection: [Ultralytics Docs](https://docs.ultralytics.com)
    - Star the repo if you find this useful!
    """,
    examples=get_example_images(),
    cache_examples=False,  # Disable caching to speed up startup and reduce storage
    allow_flagging="never",
    theme=gr.themes.Soft(),
    css=custom_css
)

# Launch the interface
interface.launch()