# Helmet Detection Project

**Link**: [Helmet Detection using YOLOv8 on Hugging Face](https://huggingface.co/spaces/Abs6187/Helmet-Detect-model)
![Helmet Detection using YOLOv8 on Hugging Face](https://github.com/Abs6187/Helmet-Detection/blob/main/Hugging-Face.png?raw=true)
![Helmet Detection Results](https://github.com/meryemsakin/helmet-detection/blob/main/allresults.jpeg)

## Project Objective

The objective of this project is to detect helmets in images and videos using the YOLOv8 object detection algorithm. The project workflow involves loading the pre-trained YOLOv8 model, resizing input frames, passing them through the model for object detection, visualizing the detections, and storing the results in annotated images and a CSV file.

### Tools Used

1. **Python Programming Language**
2. **OpenCV** - To work with images and videos
3. **YOLOv8 (You Only Look Once)** Model - For object detection
4. **Supervision** - For visualization of object detection and annotations
5. **Ultralytics** - To use the YOLO model

### Project Workflow

1. Load the pre-trained YOLOv8 model for helmet detection.
2. Resize input images or video frames.
3. Pass frames through the YOLOv8 model to detect helmets.
4. Visualize the detections using Supervision.
5. Store the resulting annotated images.
6. Generate a CSV file containing the detection labels and results.
7. Evaluate the detections using a confusion matrix.
8. Calculate accuracy and loss metrics and store the output as graphs.

### Results

| **Image**                           | **Detections**                   | **Helmet Present** | **Confidence** |
|-------------------------------------|----------------------------------|-------------------|----------------|
| ![Image 1](https://github.com/meryemsakin/helmet-detection/blob/main/Result/floor_1/images/hard_hat_workers42.png) | Worker with helmet | Yes               | 95%            |
| ![Image 2](https://github.com/meryemsakin/helmet-detection/blob/main/Result/floor_1/images/image_6.jpg) | Worker without helmet | No                | 88%            |
| ![Image 3](https://github.com/meryemsakin/helmet-detection/blob/main/Result/floor_1/images/image_7.jpg) | Worker with helmet | Yes               | 92%            |

### Metrics

#### Accuracy Graph
![Accuracy Graph](https://github.com/meryemsakin/helmet-detection/blob/main/graph.png)

#### Confusion Matrix
![Confusion Matrix](https://github.com/meryemsakin/helmet-detection/blob/main/cmatrix.png)

### Limitations and Potential Improvements

1. **False Positives/Negatives**: The model may not be accurate in all cases. Fine-tuning on a larger dataset could improve this.
2. **Additional Equipment Detection**: The project could be extended to detect other safety equipment, like gloves or safety glasses.
3. **Real-Time Detection**: Future improvements can include integrating live camera feeds for real-time detection.

### Conclusion

This project successfully detects helmets using the YOLOv8 model. The use of Python, OpenCV, YOLO, Supervision, and Ultralytics provided an efficient and accurate solution for detecting safety equipment in images and videos. Future work could enhance the model's accuracy and real-time capabilities.

