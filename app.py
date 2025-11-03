import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os

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

# Define the prediction function
def predict(image):
    results = model(image)  # Run YOLO model on the uploaded image
    results_img = results[0].plot()  # Get image with bounding boxes
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

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title=f"Helmet Detection with YOLO",
    description=f"Upload an image to detect helmets. **Currently using: {model_name}**",
    examples=get_example_images()
)

# Launch the interface
interface.launch()