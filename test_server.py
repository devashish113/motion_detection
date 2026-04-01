"""
Test script for Motion Detection Server.
Sends sample frames to the server and verifies responses.
Run the server first: python app.py
"""

import requests
import cv2
import numpy as np
import time

SERVER_URL = "http://localhost:5001"


def create_dummy_frame(color=(100, 100, 100), width=640, height=480):
    """Create a solid-color dummy frame as JPEG bytes."""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def create_frame_with_object(x, y, width=640, height=480):
    """Create a frame with a white rectangle (simulates a moving object)."""
    frame = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)
    cv2.rectangle(frame, (x, y), (x + 100, y + 100), (255, 255, 255), -1)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def test_health():
    """Test health check endpoint."""
    print("=" * 50)
    print("🔍 Testing /health endpoint...")
    resp = requests.get(f"{SERVER_URL}/health")
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
    assert resp.status_code == 200
    print("   ✅ Health check passed!\n")


def test_no_motion():
    """Send two identical frames — should detect no motion."""
    print("=" * 50)
    print("🔍 Testing NO MOTION (two identical frames)...")

    # Reset previous frame
    requests.post(f"{SERVER_URL}/reset")

    frame = create_dummy_frame(color=(100, 100, 100))

    # Frame 1 — baseline
    resp1 = requests.post(f"{SERVER_URL}/detect", data=frame,
                          headers={"Content-Type": "image/jpeg"})
    print(f"   Frame 1: {resp1.json()['message']}")

    # Frame 2 — same frame, no motion expected
    resp2 = requests.post(f"{SERVER_URL}/detect", data=frame,
                          headers={"Content-Type": "image/jpeg"})
    result = resp2.json()
    print(f"   Frame 2: {result['message']}")
    print(f"   Score: {result['score']}")
    assert result["motion"] == False, "Expected no motion!"
    print("   ✅ No motion test passed!\n")


def test_motion_detected():
    """Send two different frames — should detect motion."""
    print("=" * 50)
    print("🔍 Testing MOTION DETECTED (two different frames)...")

    # Reset previous frame
    requests.post(f"{SERVER_URL}/reset")

    # Frame 1 — dark background (baseline)
    frame1 = create_dummy_frame(color=(50, 50, 50))
    resp1 = requests.post(f"{SERVER_URL}/detect", data=frame1,
                          headers={"Content-Type": "image/jpeg"})
    print(f"   Frame 1: {resp1.json()['message']}")

    # Frame 2 — bright background (big change → motion!)
    frame2 = create_dummy_frame(color=(200, 200, 200))
    resp2 = requests.post(f"{SERVER_URL}/detect", data=frame2,
                          headers={"Content-Type": "image/jpeg"})
    result = resp2.json()
    print(f"   Frame 2: {result['message']}")
    print(f"   Score: {result['score']}")
    print(f"   Contours: {result['contours']}")
    assert result["motion"] == True, "Expected motion detected!"
    print("   ✅ Motion detection test passed!\n")


def test_moving_object():
    """Simulate an object moving across the frame."""
    print("=" * 50)
    print("🔍 Testing MOVING OBJECT simulation...")

    # Reset
    requests.post(f"{SERVER_URL}/reset")

    positions = [(50, 200), (150, 200), (300, 200), (450, 200)]

    for i, (x, y) in enumerate(positions):
        frame = create_frame_with_object(x, y)
        resp = requests.post(f"{SERVER_URL}/detect", data=frame,
                             headers={"Content-Type": "image/jpeg"})
        result = resp.json()
        motion_icon = "🔴" if result["motion"] else "⚪"
        print(f"   Frame {i + 1} (object at {x},{y}): {motion_icon} motion={result['motion']}, score={result['score']}")
        time.sleep(0.2)

    print("   ✅ Moving object test completed!\n")


def test_reset():
    """Test reset endpoint."""
    print("=" * 50)
    print("🔍 Testing /reset endpoint...")
    resp = requests.post(f"{SERVER_URL}/reset")
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
    assert resp.status_code == 200
    print("   ✅ Reset test passed!\n")


if __name__ == "__main__":
    print("\n🚀 Motion Detection Server — Test Suite")
    print("=" * 50)
    print(f"   Target: {SERVER_URL}\n")

    try:
        test_health()
        test_no_motion()
        test_motion_detected()
        test_moving_object()
        test_reset()

        print("=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)

    except requests.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running:")
        print("   python app.py")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
