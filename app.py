import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os



# Load the trained YOLOv11 model
model = YOLO("yolov11nbest.pt")

# Define the prediction function
def predict(image):
    results = model(image)  # Run YOLOv8 model on the uploaded image
    results_img = results[0].plot()  # Get image with bounding boxes
    return Image.fromarray(results_img)

# Get example images from the root folder
def get_example_images():
    examples = []
    for filename in os.listdir("."):
        if filename.startswith("Sample-Image-") and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            examples.append(filename)
    return sorted(examples)

# Create Gradio interface
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Image(type="pil"),
    title="Helmet Detection with YOLOv8",
    description="Upload an image to detect helmets.",
    examples=get_example_images()
)

# Launch the interface
interface.launch()