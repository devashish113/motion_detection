# EC2-1 — Motion Detection Server

Lightweight Flask server for detecting motion in surveillance frames using OpenCV frame differencing.

## How It Works

1. Receives JPEG frame bytes via `POST /detect`
2. Converts frame to **grayscale** + applies **Gaussian blur** (noise reduction)
3. Computes **absolute difference** between current and previous frame
4. Applies **binary threshold** — if white pixels > 2% of frame → motion detected
5. Returns JSON response with motion status, score, and contour count

## API Endpoints

| Method | Route     | Description                              |
|--------|-----------|------------------------------------------|
| POST   | `/detect` | Send JPEG frame → Get motion result      |
| GET    | `/health` | Health check (returns server status)      |
| POST   | `/reset`  | Clear stored reference frame              |

## Response Format

```json
{
  "motion": true,
  "score": 0.0823,
  "contours": 5,
  "message": "Motion DETECTED — score: 0.0823",
  "server": "motion_detection",
  "timestamp": "2026-04-01 23:30:00"
}
```

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

## EC2 Deployment

```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@<ec2-ip>

# Install deps
sudo apt update
sudo apt install python3-pip -y
pip install -r requirements.txt

# Run server (background)
nohup python3 app.py &

# Or use Docker
docker build -t motion-detection .
docker run -d -p 5001:5001 motion-detection
```

## EC2 Security Group

Open **port 5001** for inbound TCP traffic (from Lambda / API Gateway IP range).

## Tech Stack

- Python 3.11
- Flask
- OpenCV (headless)
- NumPy
