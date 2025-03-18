from flask import Flask, render_template, Response, url_for, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
import pygame
import time
from collections import deque
import random
import math
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

app = Flask(__name__, static_folder='static')

# Initialize Pygame and its mixer for audio
pygame.init()
# pygame.mixer.init()  # Commented out audio mixer

# Background music set to None since we're not using music for now
background_music = None

# Load FER2013 dataset using Kaggle SDK
def load_dataset():
    try:
        print("Loading FER2013 dataset from Kaggle...")
        dataset = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "deadskull7/fer2013",
            "fer2013.csv"
        )
        print("Dataset loaded successfully!")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have authenticated with Kaggle and have internet access.")
        return None

# Initialize dataset as None, it will be loaded when needed
fer2013_df = None

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load the pre-trained model
try:
    model.load_weights('model.h5')
    print("Successfully loaded model weights")
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise Exception("Failed to load model weights. Please ensure model.h5 is present in the correct location.")

# Dictionary which assigns each label an emotion
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Spiritual teachings and videos for each emotion
spiritual_content = {
    "Angry": {
        "text": "Breathe deeply. Let peace flow through you like a gentle stream.",
        "video": "angry.mp4"
    },
    "Disgusted": {
        "text": "Every experience teaches. Find wisdom in acceptance.",
        "video": "disgusted.mp4"
    },
    "Fearful": {
        "text": "Trust in Nirankar's presence. You are never alone.",
        "video": "neutral.mp4"
    },
    "Happy": {
        "text": "Your joy creates ripples of light. Share this blessing.",
        "video": "happy.mp4"
    },
    "Neutral": {
        "text": "Let Nirankar's light fill your heart, smile to create ripples.",
        "video": "neutral.mp4"
    },
    "Sad": {
        "text": "This too shall pass. Feel Nirankar's comfort embrace you.",
        "video": "neutral.mp4"
    },
    "Surprised": {
        "text": "Each moment reveals Nirankar's wonder. Stay open.",
        "video": "surprised.mp4"
    }
}

# Try to load face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Failed to load cascade classifier")
    print("Successfully loaded face cascade classifier")
except Exception as e:
    print(f"Error loading face cascade classifier: {e}")
    raise Exception("Failed to load face cascade classifier. Please ensure haarcascade_frontalface_default.xml is present in the correct location.")

class RippleEffect:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.damping = 0.95  # Even longer-lasting ripples
        self.current_buffer = np.zeros((height, width), dtype=np.float32)
        self.previous_buffer = np.zeros((height, width), dtype=np.float32)
        self.ripple_points = []
        self.emotion_colors = {
            "Happy": (0, 255, 255),    # Cyan for happiness
            "Surprised": (255, 215, 0), # Gold for surprise
            "Neutral": (200, 200, 200), # Silver for neutral
            "Sad": (0, 0, 255),        # Blue for sadness
            "Angry": (255, 0, 0),      # Red for anger
            "Fearful": (128, 0, 128),  # Purple for fear
            "Disgusted": (0, 128, 0)   # Green for disgust
        }
        
    def create_emotion_ripples(self, x, y, emotion, strength=1.0):
        """Create emotion-specific ripples"""
        base_radius = 60 if emotion in ["Happy", "Surprised"] else 40
        num_ripples = 5 if emotion in ["Happy", "Surprised"] else 3
        
        # Create concentric ripples
        for i in range(num_ripples):
            radius = base_radius * (i + 1) * 0.8
            self.create_drop(x, y, radius, strength * (1 - i*0.15), emotion)
            
            # Add satellite ripples for positive emotions
            if emotion in ["Happy", "Surprised"]:
                angle = 2 * math.pi * i / num_ripples
                offset_x = x + int(radius * 0.5 * math.cos(angle))
                offset_y = y + int(radius * 0.5 * math.sin(angle))
                self.create_drop(offset_x, offset_y, radius * 0.5, strength * 0.7, emotion)
        
    def create_drop(self, x, y, radius, strength, emotion):
        """Create a drop with emotion-specific effects"""
        x, y = int(x), int(y)
        radius = int(radius)
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Add ripple point with color information
        self.ripple_points.append({
            'x': x, 'y': y,
            'radius': radius,
            'strength': strength,
            'color': color,
            'emotion': emotion,
            'age': 0,
            'phase': 0
        })
        
        # Create ripple disturbance
        y_min, y_max = max(0, y - radius), min(self.height, y + radius)
        x_min, x_max = max(0, x - radius), min(self.width, x + radius)
        
        for i in range(y_min, y_max):
            for j in range(x_min, x_max):
                distance = math.sqrt((i - y) ** 2 + (j - x) ** 2)
                if distance < radius:
                    impact = strength * (1 - distance/radius)
                    # Add wave pattern
                    wave = math.sin(distance/radius * math.pi * 2)
                    self.previous_buffer[i, j] += impact * wave
    
    def update(self):
        """Update ripple simulation with enhanced effects"""
        # Update wave propagation
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                val = (
                    self.previous_buffer[i-1, j] + 
                    self.previous_buffer[i+1, j] +
                    self.previous_buffer[i, j-1] + 
                    self.previous_buffer[i, j+1] +
                    self.previous_buffer[i-1, j-1] * 0.7 +
                    self.previous_buffer[i-1, j+1] * 0.7 +
                    self.previous_buffer[i+1, j-1] * 0.7 +
                    self.previous_buffer[i+1, j+1] * 0.7
                ) / 5.8 - self.current_buffer[i, j]
                
                self.current_buffer[i, j] = val * self.damping
        
        # Swap buffers
        self.previous_buffer, self.current_buffer = self.current_buffer, self.previous_buffer
        
        # Update ripple points
        for point in self.ripple_points[:]:
            point['age'] += 1
            point['phase'] = (point['phase'] + 0.1) % (2 * math.pi)
            if point['age'] > 60:  # Longer lifetime for ripples
                self.ripple_points.remove(point)

class GameState:
    def __init__(self):
        self.start_time = None
        self.countdown_start = None
        self.countdown_duration = 5.0  # 5 second countdown
        self.last_capture_time = 0
        self.capture_interval = 2.0
        self.emotion_weights = {emotion: 0 for emotion in ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]}
        self.current_emotion = None
        self.last_emotion_change = 0
        self.emotion_stability_threshold = 3
        self.teaching_display_time = None
        self.teaching_duration = 5.0
        self.session_active = False
        self.session_duration = 45.0
        self.ripple_effect = None
        self.state = "WAITING"  # WAITING, COUNTDOWN, ACTIVE, COMPLETE
        self.emotion_update = None  # Store the latest emotion update
        
game_state = GameState()

# Initialize camera only if not in production
if os.environ.get('FLASK_ENV') != 'production':
    for camera_index in [1, 0, 2]:
        camera = cv2.VideoCapture(camera_index)
        if camera.isOpened():
            ret, frame = camera.read()
            if ret:
                camera.release()
                camera = cv2.VideoCapture(camera_index)
                print(f"Successfully connected to camera {camera_index}")
                break
            camera.release()
    else:
        print("Error: Could not find working camera")
        camera = cv2.VideoCapture(0)
else:
    print("Running in production mode - camera initialization skipped")
    camera = None

def update_emotion_weights(detected_emotion):
    """Update emotion weights with decay for others"""
    decay_factor = 0.7
    for emotion in game_state.emotion_weights:
        if emotion == detected_emotion:
            game_state.emotion_weights[emotion] += 1
        else:
            game_state.emotion_weights[emotion] *= decay_factor
    
    # Get the most confident emotion
    max_weight = max(game_state.emotion_weights.values())
    if max_weight >= game_state.emotion_stability_threshold:
        dominant_emotion = max(game_state.emotion_weights.items(), key=lambda x: x[1])[0]
        if dominant_emotion != game_state.current_emotion:
            game_state.current_emotion = dominant_emotion
            game_state.teaching_display_time = time.time()
            game_state.emotion_weights = {emotion: 0 for emotion in game_state.emotion_weights}
            
            # Return the spiritual content for the new emotion
            return {
                "emotion": dominant_emotion,
                "teaching": spiritual_content[dominant_emotion]["text"],
                "video_path": f'videos/{spiritual_content[dominant_emotion]["video"]}'
            }
    return None

def create_ripple_surface(frame, ripple_effect):
    """Create enhanced ripple effect with emotion-specific visuals"""
    height, width = frame.shape[:2]
    if game_state.ripple_effect is None:
        game_state.ripple_effect = RippleEffect(width, height)
    
    # Update ripple simulation
    ripple_effect.update()
    
    # Create displacement map
    displacement = ripple_effect.current_buffer
    
    # Create coordinate matrices
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Apply stronger displacement for more visible effect
    displacement_strength = 12.0  # Increased displacement
    x_displaced = x_coords + displacement * displacement_strength
    y_displaced = y_coords + displacement * displacement_strength
    
    # Ensure coordinates are within bounds
    x_displaced = np.clip(x_displaced, 0, width-1).astype(np.int32)
    y_displaced = np.clip(y_displaced, 0, height-1).astype(np.int32)
    
    # Create the displaced frame
    frame_displaced = frame.copy()
    frame_displaced = frame[y_displaced, x_displaced]
    
    # Add emotion-specific color effects
    ripple_intensity = np.abs(displacement)
    max_intensity = np.max(ripple_intensity)
    if max_intensity > 0:
        ripple_intensity = ripple_intensity / max_intensity
    
    # Create emotion-specific color overlay
    if game_state.current_emotion:
        color = np.array(ripple_effect.emotion_colors[game_state.current_emotion]) / 255.0
        color_overlay = np.zeros_like(frame, dtype=np.float32)
        
        for c in range(3):
            color_overlay[:,:,c] = ripple_intensity * color[c]
        
        # Apply color overlay with enhanced visibility
        alpha = ripple_intensity * 0.8  # Increased visibility
        alpha = np.stack([alpha] * 3, axis=-1)
        frame_displaced = frame_displaced * (1 - alpha) + (color_overlay * 255) * alpha
        
        # Add glow effect for positive emotions
        if game_state.current_emotion in ["Happy", "Surprised"]:
            glow = cv2.GaussianBlur(color_overlay, (21, 21), 0)
            frame_displaced = cv2.addWeighted(frame_displaced, 1, (glow * 255).astype(np.uint8), 0.5, 0)
    
    return frame_displaced.astype(np.uint8)

def generate_frames():
    if camera is None:
        return b''
        
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            break
        
        current_time = time.time()
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            # State machine for game flow
            if game_state.state == "WAITING" and len(faces) > 0:
                game_state.state = "COUNTDOWN"
                game_state.countdown_start = current_time
                
            elif game_state.state == "COUNTDOWN":
                remaining = game_state.countdown_duration - (current_time - game_state.countdown_start)
                if remaining > 0:
                    # Display countdown
                    cv2.putText(frame, f"Starting in: {int(remaining)}s", 
                              (frame.shape[1]//2 - 100, frame.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    game_state.state = "ACTIVE"
                    game_state.start_time = current_time
                    game_state.session_active = True
            
            elif game_state.state == "ACTIVE":
                # Process faces and emotions
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    
                    if current_time - game_state.last_capture_time >= game_state.capture_interval:
                        roi_gray = gray[y:y + h, x:x + w]
                        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                        prediction = model.predict(cropped_img)
                        detected_emotion = emotion_dict[int(np.argmax(prediction))]
                        emotion_update = update_emotion_weights(detected_emotion)
                        game_state.last_capture_time = current_time
                        
                        # If there's a new emotion, store it for the /emotion_update endpoint
                        if emotion_update:
                            game_state.emotion_update = emotion_update
                        
                        # Create new ripples based on emotion
                        center_x, center_y = x + w//2, y + h//2
                        if game_state.ripple_effect and game_state.current_emotion:
                            strength = 1.5 if game_state.current_emotion in ["Happy", "Surprised"] else 1.0
                            game_state.ripple_effect.create_emotion_ripples(
                                center_x, center_y,
                                game_state.current_emotion,
                                strength=strength
                            )
                    
                    # Display emotion and teaching
                    if game_state.current_emotion:
                        cv2.putText(frame, game_state.current_emotion, (x+20, y-60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        if game_state.teaching_display_time:
                            teaching_age = current_time - game_state.teaching_display_time
                            if teaching_age < game_state.teaching_duration:
                                alpha = 1.0 - (teaching_age / game_state.teaching_duration)
                                teaching = spiritual_content[game_state.current_emotion]["text"]
                                cv2.putText(frame, teaching, (10, frame.shape[0] - 20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                                          max(1, int(2 * alpha)), cv2.LINE_AA)
                
                # Apply ripple effect to frame
                if game_state.ripple_effect:
                    frame = create_ripple_surface(frame, game_state.ripple_effect)
                
                # Check session completion
                if current_time - game_state.start_time >= game_state.session_duration:
                    game_state.state = "COMPLETE"
                    
            elif game_state.state == "COMPLETE":
                final_message = "Session complete. Take a moment to reflect."
                cv2.putText(frame, final_message, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                game_state.session_active = False
                game_state.emotion_weights = {emotion: 0 for emotion in game_state.emotion_weights}
                game_state.current_emotion = None
                game_state.state = "WAITING"
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error processing frame: {e}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_update')
def emotion_update():
    """Endpoint to check for new emotion updates"""
    if game_state.emotion_update:
        data = game_state.emotion_update
        # Clear the update after sending
        game_state.emotion_update = None
        return jsonify(data)
    return jsonify({"emotion": None})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Make sure videos directory exists
    os.makedirs(os.path.join('static', 'videos'), exist_ok=True)
    
    # Create the HTML template with updated JavaScript
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Detection Game - Sant Nirankari Mission</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .video-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            position: relative;
        }
        img {
            max-width: 100%;
            border-radius: 5px;
        }
        .error-message {
            display: none;
            color: red;
            margin-top: 20px;
        }
        #videoFeed {
            min-height: 300px;
        }
        .instructions {
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            z-index: 1000;
        }
        .modal-content {
            position: relative;
            margin: auto;
            width: 90%;
            max-width: 800px;
            top: 50%;
            transform: translateY(-50%);
        }
        .teaching-video {
            width: 100%;
            height: auto;
            max-height: 80vh;
        }
        .teaching-text {
            color: white;
            text-align: center;
            margin-top: 20px;
            font-size: 1.2em;
        }
    </style>
    <script>
        let currentVideoTimeout = null;
        let lastEmotion = null;
        let emotionCheckInterval = null;
        
        function showTeaching(videoPath, text, duration = 5000) {
            const modal = document.getElementById('teachingModal');
            const video = document.getElementById('teachingVideo');
            const teachingText = document.getElementById('teachingText');
            
            // Clear any existing timeout
            if (currentVideoTimeout) {
                clearTimeout(currentVideoTimeout);
            }
            
            // Set up video and text
            video.src = '/static/' + videoPath;
            teachingText.textContent = text;
            
            // Show modal
            modal.style.display = 'block';
            
            // Play video
            video.play().catch(function(error) {
                console.log("Video play failed:", error);
            });
            
            // Set timeout to hide modal
            currentVideoTimeout = setTimeout(() => {
                hideTeaching();
            }, duration);
        }
        
        function hideTeaching() {
            const modal = document.getElementById('teachingModal');
            const video = document.getElementById('teachingVideo');
            
            // Stop video and hide modal
            video.pause();
            video.currentTime = 0;
            modal.style.display = 'none';
        }
        
        function checkForEmotionUpdates() {
            fetch('/emotion_update')
                .then(response => response.json())
                .then(data => {
                    if (data.emotion && data.emotion !== lastEmotion) {
                        lastEmotion = data.emotion;
                        showTeaching(data.video_path, data.teaching, 5000);
                    }
                })
                .catch(error => {
                    console.error('Error checking for emotion updates:', error);
                });
        }
        
        window.onload = function() {
            const img = document.getElementById('videoFeed');
            const errorMsg = document.getElementById('errorMessage');
            
            img.onerror = function() {
                errorMsg.style.display = 'block';
                img.style.display = 'none';
            };
            
            img.onload = function() {
                errorMsg.style.display = 'none';
                img.style.display = 'block';
            };
            
            // Start polling for emotion updates
            emotionCheckInterval = setInterval(checkForEmotionUpdates, 500);
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>Ripple Effect - A Mindful Journey</h1>
        <div class="instructions">
            <h3>Welcome to the Ripple Effect Game</h3>
            <p>Create waves of positivity through mindful emotional awareness.</p>
            <p>Each session lasts 45 seconds. Listen to the teachings and let your emotions flow naturally.</p>
        </div>
        <div class="video-container">
            <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video feed">
            <div id="errorMessage" class="error-message">
                Camera not available. Please check your camera settings and make sure:
                <ul style="text-align: left;">
                    <li>Your camera is enabled</li>
                    <li>No other application is using the camera</li>
                    <li>Camera permissions are granted to your browser</li>
                </ul>
            </div>
        </div>
    </div>
    
    <!-- Teaching Modal -->
    <div id="teachingModal" class="modal">
        <div class="modal-content">
            <video id="teachingVideo" class="teaching-video" playsinline>
                Your browser does not support the video tag.
            </video>
            <div id="teachingText" class="teaching-text"></div>
        </div>
    </div>
</body>
</html>
        ''')
    
    # Get port from environment variable for production deployment
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting Ripple Effect Game. Please open http://127.0.0.1:{port} in your browser")
    app.run(host='0.0.0.0', port=port, debug=debug)