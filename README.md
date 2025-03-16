# Ripple Effect - Emotion Detection Game

A mindful journey through emotional awareness, creating visual ripples based on detected emotions.

## Overview

Ripple Effect is an interactive web application that uses computer vision and deep learning to detect emotions from facial expressions in real-time. The application creates beautiful ripple effects that vary based on the detected emotion, providing a unique visualization of emotional states.

## Features

- Real-time emotion detection from webcam feed
- Dynamic ripple effects based on emotions
- Spiritual teachings for each emotional state
- Interactive session-based gameplay
- Beautiful visual effects for different emotions

## Prerequisites

- Python 3.10 or higher
- Webcam access
- Kaggle account and API credentials (for dataset download)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/shrutipanjwani/ripple-effect.git
cd ripple-effect
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up Kaggle credentials:
   - Create a Kaggle account if you don't have one
   - Go to your Kaggle account settings (https://www.kaggle.com/account)
   - Create a new API token (this will download kaggle.json)
   - Place the kaggle.json file in one of these locations:
     - Linux/macOS: ~/.kaggle/kaggle.json
     - Windows: C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json

## Dataset

The application uses the FER2013 dataset for emotion detection. The dataset will be automatically downloaded when you run the application for the first time using the Kaggle API.

## Running the Application

1. Navigate to the src directory:

```bash
cd src
```

2. Run the application:

```bash
python web_app.py
```

3. Open your web browser and go to:

```
http://localhost:5000
```

## How to Play

1. Allow camera access when prompted
2. Position yourself in front of the camera
3. The game will start automatically when it detects your face
4. Express different emotions and watch as they create unique ripple effects
5. Each session lasts 45 seconds
6. Read and reflect on the spiritual teachings that appear with each emotion

## Emotions Detected

- Happy (Cyan ripples)
- Surprised (Gold ripples)
- Neutral (Silver ripples)
- Sad (Blue ripples)
- Angry (Red ripples)
- Fearful (Purple ripples)
- Disgusted (Green ripples)

## Technical Details

- Built with Flask for the web framework
- Uses OpenCV for face detection
- TensorFlow/Keras for emotion recognition
- Custom ripple effect algorithm for visualizations
- Kaggle SDK for dataset management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FER2013 dataset from Kaggle
- Sant Nirankari Mission for spiritual teachings
- OpenCV and TensorFlow communities
