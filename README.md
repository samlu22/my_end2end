**Video Object Detection and Team Classification**

This project uses the YOLO object detection model from Ultralytics to process a video, identify people, and classify them into teams. The script outputs annotated frames with bounding boxes, team labels and more.

🚀 Features
Real-time Object Detection: Processes video frames to detect objects, specifically people, using a custom-trained YOLO model.

Team Classification: Analyzes the color of detected people's jerseys to classify them as either on the "Red Team" or "Blue Team."

Annotated Output: Saves individual frames with drawn bounding boxes, confidence scores, and team labels to a specified output directory.

Frame Processing: Includes logic to resize video frames to a standard 720p height for consistent processing.

📦 Dependencies
To run this script, you need to install the following Python libraries:

Bash

pip install opencv-python ultralytics numpy


**🧠 Model Source**
The model used in this project is a custom-trained YOLO model provided as bset.pt. The .pt extension stands for "PyTorch" and is a standard format for saving model weights. The file best.pt is a common naming convention used by the Ultralytics framework to denote the checkpoint that achieved the best performance during the training of the model. This is not a pre-trained model from a public dataset but a model specifically trained for your use case.

**💡 Usage**
Place your video file (e.g. extract_45s-57s.mp4) and the custom model file (best.pt) in the models directory.

Run the script from your terminal:

Bash

python main.py
The script will process the first 300 frames of the video and save the annotated images in a newly created directory called output_frames.

⚙️ How It Works
The script initializes a YOLO model using the bset.pt file.

It reads the input video frame by frame.

For each frame, it runs the YOLO model to detect objects.

It filters the detections to only consider objects labeled as 'person'.

For each detected person, it crops the bounding box region and analyzes the average color of the central area to determine.

Based on the color analysis, it draws a bounding box around the person with a corresponding color.

Finally, the annotated frame is saved as a .jpg image in the output_frames directory.
