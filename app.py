"""
EC2-1 — Motion Detection Server
Flask server that detects motion using OpenCV frame differencing.
Compares current frame against previous frame, flags motion above threshold.
Runs on port 5001.
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ─── Store previous frame for comparison ───
previous_frame = None

# ─── Configuration ───
MOTION_THRESHOLD = 25        # Pixel intensity threshold for binary diff
MIN_MOTION_AREA = 0.02       # Minimum % of frame that must change (2%)
GAUSSIAN_BLUR_SIZE = (21, 21) # Blur kernel size to reduce noise


def detect_motion(current_frame_bytes):
    """
    Detect motion by comparing current frame with previous frame.
    Uses grayscale conversion + Gaussian blur + absolute difference + thresholding.
    
    Returns:
        dict: {motion: bool, score: float, contours: int, message: str}
    """
    global previous_frame

    # Decode JPEG bytes to numpy array
    nparr = np.frombuffer(current_frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "motion": False,
            "score": 0.0,
            "contours": 0,
            "message": "Failed to decode frame"
        }

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_SIZE, 0)

    # If no previous frame, store this one and return no motion
    if previous_frame is None:
        previous_frame = gray
        return {
            "motion": False,
            "score": 0.0,
            "contours": 0,
            "message": "First frame — stored as reference. No motion detected yet."
        }

    # Compute absolute difference between current and previous frame
    frame_diff = cv2.absdiff(previous_frame, gray)

    # Apply binary threshold — pixels above threshold become white (255)
    _, thresh = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps in detected edges
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of motion regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate motion score: percentage of frame with white pixels
    total_pixels = gray.shape[0] * gray.shape[1]
    white_pixels = cv2.countNonZero(thresh)
    motion_score = round(white_pixels / total_pixels, 4)

    # Count significant contours (filter out small noise)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 500]

    # Determine if motion is detected
    motion_detected = motion_score >= MIN_MOTION_AREA

    # Update previous frame for next comparison
    previous_frame = gray

    return {
        "motion": motion_detected,
        "score": motion_score,
        "contours": len(significant_contours),
        "message": f"Motion {'DETECTED' if motion_detected else 'not detected'} — score: {motion_score}"
    }


# ─── API Routes ───

@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    Receives JPEG frame bytes in request body.
    Returns motion detection result as JSON.
    """
    try:
        # Get raw frame bytes from request body
        frame_bytes = request.get_data()

        if not frame_bytes:
            return jsonify({
                "error": "No frame data received",
                "motion": False,
                "score": 0.0
            }), 400

        # Run motion detection
        result = detect_motion(frame_bytes)

        # Add metadata
        result["server"] = "motion_detection"
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "motion": False,
            "score": 0.0,
            "server": "motion_detection"
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "server": "motion_detection",
        "port": 5001,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }), 200


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the previous frame reference (start fresh)."""
    global previous_frame
    previous_frame = None
    return jsonify({
        "message": "Previous frame reference cleared. Next frame will be used as new baseline.",
        "server": "motion_detection"
    }), 200


if __name__ == "__main__":
    print("=" * 50)
    print("🟢 EC2-1 — Motion Detection Server")
    print("=" * 50)
    print(f"📡 Running on port 5001")
    print(f"🔗 POST /detect  → Send JPEG frame for motion detection")
    print(f"🔗 GET  /health  → Health check")
    print(f"🔗 POST /reset   → Reset frame reference")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5001, debug=False)
