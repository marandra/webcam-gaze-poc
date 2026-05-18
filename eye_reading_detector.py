"""
Eye Movement Reading Detector
Tracks eye movements via webcam to detect reading patterns using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from scipy import signal

class EyeReadingDetector:
    def __init__(self, history_size=90, fps=30): # A 3-second window
        """
        Initialize the Eye Reading Detector

        Args:
            history_size: Number of frames to keep in history for analysis
            fps: Expected frames per second for frequency calculation
        """
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmark indices
        # Left eye landmarks (from viewer's perspective)
        self.LEFT_EYE_INNER = 133
        self.LEFT_EYE_OUTER = 33
        self.LEFT_IRIS = 468  # Center of left iris

        # Right eye landmarks
        self.RIGHT_EYE_INNER = 362
        self.RIGHT_EYE_OUTER = 263
        self.RIGHT_IRIS = 473  # Center of right iris
        
        # Eye contour indices for visualization
        self.LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

        # Movement tracking
        self.history_size = history_size
        self.fps = fps
        self.timestamps = deque(maxlen=history_size)
        self.normalized_positions = deque(maxlen=history_size) # Replaced separate deques with a single one for the combined signal

        # Reading detection parameters
        self.reading_freq_min = 0.5  # Hz
        self.reading_freq_max = 4.0  # Hz
        self.movement_threshold = 0.02 # Min stdev of normalized signal to start analysis

        # MODIFICATION: Pre-design a band-pass filter for efficiency
        nyquist = 0.5 * self.fps
        low = self.reading_freq_min / nyquist
        high = self.reading_freq_max / nyquist
        # A 4th-order Butterworth filter is a good choice
        self.b, self.a = signal.butter(4, [low, high], btype='band')

        # MODIFICATION: Add a buffer for temporal smoothing (hysteresis)
        self.reading_buffer_len = 10 # Check the last 10 analyses
        self.reading_buffer = deque(maxlen=self.reading_buffer_len)
        self.reading_confirm_threshold = 0.7 # 70% of recent frames must be 'reading' to confirm

        # Status variables
        self.is_reading = False
        self.dominant_frequency = 0.0
        self.confidence = 0.0
        
    def calculate_relative_position(self, iris_point, inner_corner, outer_corner):
        """
        Calculate normalized horizontal position of iris relative to eye corners
        
        Args:
            iris_point: (x, y) coordinates of iris center
            inner_corner: (x, y) coordinates of inner eye corner
            outer_corner: (x, y) coordinates of outer eye corner
            
        Returns:
            Normalized position (-1 to 1, where -1 is outer, 1 is inner)
        """
        # Calculate eye width
        eye_vector = np.array(inner_corner) - np.array(outer_corner)
        iris_vector = np.array(iris_point) - np.array(outer_corner)
        eye_width_sq = np.dot(eye_vector, eye_vector)
        if eye_width_sq == 0:
            return 0
        # Calculate projection
        projection = np.dot(iris_vector, eye_vector) / eye_width_sq

        # Normalize to [-1, 1]
        return np.clip((projection * 2) - 1, -1.5, 1.5) # Clip to handle edge cases

    # MODIFICATION: This function is completely replaced with the new robust method
    def analyze_movement_pattern(self):
        """
        Analyze eye movement patterns using a band-pass filter and Welch's method.
        Returns:
            (is_reading, dominant_frequency, confidence)
        """
        if len(self.normalized_positions) < self.history_size:
            return False, 0.0, 0.0

        signal_data = np.array(list(self.normalized_positions))

        # 1. Check if there's enough movement to analyze
        if np.std(signal_data) < self.movement_threshold:
            return False, 0.0, 0.0

        # 2. Apply the pre-designed band-pass filter to isolate reading frequencies
        filtered_signal = signal.lfilter(self.b, self.a, signal_data)

        # 3. Use Welch's method for robust Power Spectral Density (PSD) estimation
        # This is more stable than a single FFT. nperseg is the segment length.
        freqs, psd = signal.welch(filtered_signal, self.fps, nperseg=self.history_size // 2)

        # 4. Analyze the power within the target frequency band
        reading_band_indices = np.where((freqs >= self.reading_freq_min) & (freqs <= self.reading_freq_max))
        
        if len(reading_band_indices[0]) == 0:
            return False, 0.0, 0.0 # No frequency data in our band

        # Find the dominant frequency within the reading band
        peak_idx_in_band = np.argmax(psd[reading_band_indices])
        peak_freq_idx = reading_band_indices[0][peak_idx_in_band]
        dominant_freq = freqs[peak_freq_idx]

        # 5. Calculate a robust confidence score
        # Confidence = Power within the reading band / Total power in the spectrum
        power_in_band = np.sum(psd[reading_band_indices])
        total_power = np.sum(psd)
        
        if total_power < 1e-8: # Avoid division by zero
            return False, 0.0, 0.0
            
        confidence = power_in_band / total_power

        # 6. Make the preliminary reading decision
        # The main condition is high confidence. The frequency is already constrained by the filter.
        is_reading_now = confidence > 0.6  # Threshold can be tuned

        return is_reading_now, dominant_freq, confidence

    def draw_eye_tracking(self, image, landmarks):
        """
        Draw eye landmarks and tracking visualization on the image
        
        Args:
            image: The image to draw on
            landmarks: Face mesh landmarks
        """
        h, w = image.shape[:2]
        
        # Draw eye contours
        for idx in self.LEFT_EYE_CONTOUR:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        for idx in self.RIGHT_EYE_CONTOUR:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # Draw eye corners (larger, different color)
        corner_indices = [self.LEFT_EYE_INNER, self.LEFT_EYE_OUTER, 
                         self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER]
        for idx in corner_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
        
        # Draw iris centers (if available)
        if len(landmarks) > 468:  # Check if iris landmarks are available
            # Left iris
            x = int(landmarks[self.LEFT_IRIS].x * w)
            y = int(landmarks[self.LEFT_IRIS].y * h)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(image, (x, y), 8, (0, 0, 255), 1)
            
            # Right iris
            x = int(landmarks[self.RIGHT_IRIS].x * w)
            y = int(landmarks[self.RIGHT_IRIS].y * h)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(image, (x, y), 8, (0, 0, 255), 1)
    
    def draw_status_overlay(self, image):
        """
        Draw status information on the image
        
        Args:
            image: The image to draw on
        """
        h, w = image.shape[:2]
        
        # Create semi-transparent overlay
        overlay = image.copy()
        
        # Background for text
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Draw status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Reading status
        status_text = "READING" if self.is_reading else "NOT READING"
        status_color = (0, 255, 0) if self.is_reading else (0, 100, 255)
        cv2.putText(image, status_text, (20, 40), font, 0.8, status_color, 2)
        
        # Frequency information
        cv2.putText(image, f"Frequency: {self.dominant_frequency:.2f} Hz", 
                   (20, 70), font, 0.6, (255, 255, 255), 1)
        
        # Confidence
        cv2.putText(image, f"Confidence: {self.confidence:.1%}", 
                   (20, 95), font, 0.6, (255, 255, 255), 1)
        
        # Draw frequency visualization
        if len(self.normalized_positions) > 10:
            # Create mini graph
            graph_w, graph_h = 200, 80
            graph = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)
            
            # Plot recent positions
            recent_positions = list(self.normalized_positions)[-50:]
            if len(recent_positions) > 1:
                # Normalize positions
                positions = np.array(recent_positions)
                positions = (positions - np.min(positions)) / (np.max(positions) - np.min(positions) + 0.001)
                positions = (1 - positions) * (graph_h - 10) + 5
                
                # Draw graph
                for i in range(1, len(positions)):
                    x1 = int((i-1) * graph_w / len(positions))
                    x2 = int(i * graph_w / len(positions))
                    y1 = int(positions[i-1])
                    y2 = int(positions[i])
                    cv2.line(graph, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Place graph on main image
            image[h-graph_h-10:h-10, w-graph_w-10:w-10] = graph
    

    def process_frame(self, frame):
        """
        Process a single frame for eye tracking and reading detection.
        
        Args:
            frame: The input frame from webcam
            
        Returns:
            Processed frame with visualization
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # Process landmarks if face detected
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Draw eye tracking visualization
            self.draw_eye_tracking(frame, face_landmarks.landmark)
            
            # Extract iris positions if available
            if len(face_landmarks.landmark) > self.RIGHT_IRIS:
                # Get landmark coordinates
                h, w, _ = frame.shape

                # Extract landmark coordinates (your code is good)
                left_iris_pt = (face_landmarks.landmark[self.LEFT_IRIS].x * w,
                              face_landmarks.landmark[self.LEFT_IRIS].y * h)
                left_inner_pt = (face_landmarks.landmark[self.LEFT_EYE_INNER].x * w,
                               face_landmarks.landmark[self.LEFT_EYE_INNER].y * h)
                left_outer_pt = (face_landmarks.landmark[self.LEFT_EYE_OUTER].x * w,
                               face_landmarks.landmark[self.LEFT_EYE_OUTER].y * h)
                right_iris_pt = (face_landmarks.landmark[self.RIGHT_IRIS].x * w, face_landmarks.landmark[self.RIGHT_IRIS].y * h)
                right_inner_pt = (face_landmarks.landmark[self.RIGHT_EYE_INNER].x * w, face_landmarks.landmark[self.RIGHT_EYE_INNER].y * h)
                right_outer_pt = (face_landmarks.landmark[self.RIGHT_EYE_OUTER].x * w, face_landmarks.landmark[self.RIGHT_EYE_OUTER].y * h)

                # Calculate normalized positions
                left_pos = self.calculate_relative_position(left_iris_pt, left_inner_pt, left_outer_pt)
                right_pos = self.calculate_relative_position(right_iris_pt, right_inner_pt, right_outer_pt)

                # MODIFICATION: Use a single, averaged signal
                combined_pos = (left_pos + right_pos) / 2.0
                self.normalized_positions.append(combined_pos)
                self.timestamps.append(time.time())

                # Analyze movement and get preliminary status
                is_reading_now, freq, conf = self.analyze_movement_pattern()
                self.dominant_frequency = freq
                self.confidence = conf

                # MODIFICATION: Apply temporal smoothing
                self.reading_buffer.append(is_reading_now)
                if len(self.reading_buffer) == self.reading_buffer_len:
                    # Check if the buffer is filled enough with 'True' values
                    if sum(self.reading_buffer) >= self.reading_buffer_len * self.reading_confirm_threshold:
                        self.is_reading = True
                    else:
                        self.is_reading = False

        self.draw_status_overlay(frame)

        return frame

    def run(self):
        """
        Main loop to run the eye tracking application
        """
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Eye Reading Detector Started")
        print("Press 'q' to quit")
        print("Press 'r' to reset calibration")
        print("\nDetecting reading patterns based on eye movement frequency...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('Eye Reading Detector', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset history
                self.normalized_positions.clear()
                self.timestamps.clear()
                print("History reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()


def main():
    """
    Main function to run the application
    """
    print("Initializing Eye Reading Detector...")
    print("This application tracks eye movements to detect reading patterns.")
    print("\nRequirements:")
    print("- Good lighting conditions")
    print("- Face clearly visible to camera")
    print("- Stable head position for best results")
    print("\nStarting in 3 seconds...\n")
    
    time.sleep(3)
    
    detector = EyeReadingDetector(history_size=60, fps=30)
    detector.run()


if __name__ == "__main__":
    main()
