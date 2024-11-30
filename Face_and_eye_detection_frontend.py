import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Load Haar Cascade files
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Function for face and eye detection
def detect_faces_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # Detect faces
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
        # Write 'Face' above the rectangle
        text_position = (x, y - 10)  # Position above the rectangle
        cv2.putText(frame, 'Face', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]  # Region of interest in grayscale
        roi_color = frame[y:y+h, x:x+w]  # Region of interest in color
        
        # Detect eyes in the face region
        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 4)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # Write 'Eye' above the rectangle
            text_position = (ex, ey - 10)  # Position above the rectangle
            cv2.putText(roi_color, 'Eye', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
    return frame

# Streamlit UI
st.title("Face and Eye Recognition👀")
st.write("This application allows you to detect faces and eyes from your webcam or uploaded pictures.")

# Add custom CSS for background image
def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://image.freepik.com/free-vector/face-recognition-low-poly-wireframe-banner-template-futuristic-computer-technology-smart-identification-system-poster-polygonal-design-facial-scan-3d-mesh-art-with-connected-dots_201274-4.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Streamlit UI
background_image_url = "https://via.placeholder.com/1600x900.png?text=Background+Image"  # Replace with your image URL
add_background_image(background_image_url)

# Option to select mode
mode = st.radio("Choose an option:", ("Live Webcam Detection", "Upload and Analyze Picture"))

if mode == "Live Webcam Detection":
    start_detection = st.button("Start Webcam Detection")

    if start_detection:
        st.write("Press 'q' to stop the video stream.")
        
        # Start webcam using IVCam (index 1 or higher depending on your system)
        video_capture = cv2.VideoCapture(1)  # IVCam is usually detected as index 1, change if necessary
        stframe = st.empty()  # Placeholder for video frames
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Unable to access webcam. Please check your camera.")
                break
            # Perform face and eye detection
            frame = detect_faces_and_eyes(frame)
            # Convert the frame to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_column_width=True)
            # Check if 'q' is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()

elif mode == "Upload and Analyze Picture":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detecting faces and eyes...")
        
        # Convert the image to OpenCV format
        frame = np.array(image)
        if frame.ndim == 3 and frame.shape[2] == 4:  # Check for alpha channel
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.ndim == 3:  # Standard RGB image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Perform detection
        detected_frame = detect_faces_and_eyes(frame)
        
        # Convert the result back to RGB for Streamlit
        detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
        st.image(detected_frame, caption="Processed Image with Detections", use_column_width=True)
        st.write("Thanks for using the Face and Eye Detection App!")
