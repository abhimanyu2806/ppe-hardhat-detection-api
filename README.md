# PPE Safety Detection API

RT-DETR based hard-hat detection with a lightweight rule-based reasoning layer for answering natural-language questions about PPE compliance in images.

Built for: Pre-Hackathon Screening, Round 1 — Constrained Object Detection & Reasoning API.

---

## What this does

- **`/detect`** — accepts an image, returns bounding boxes, class names (`Hardhat` / `NO-Hardhat`), and confidence scores from a fine-tuned RT-DETR model, plus an aggregated safety assessment.
- **`/ask`** — accepts an image + a natural-language question. Decides whether the question needs detection at all (intent routing), runs detection + reasoning if so, and explicitly says "insufficient information" rather than guessing when confidence is too low.

No agentic frameworks (LangChain, CrewAI, AutoGen, etc.) are used anywhere — both endpoints use hand-written, deterministic logic.

---

## Project structure

```
Project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, both endpoints
│   ├── detector.py       # RT-DETR wrapper
│   └── reasoning.py      # Rule-based safety analysis
├── model/
│   └── best.pt            # Trained checkpoint (see "Getting the model weights" below)
├── tests/
│   ├── test_api.py
│   ├── test_reasoning.py
│   ├── test_model.py
│   └── sample_image.jpg
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repo and create a virtual environment:**
```bash
git clone <your-repo-url>
cd Project
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Get the model weights:**

The trained checkpoint (`best.pt`, ~250MB) is not stored directly in this repo due to GitHub's file size limits. Download it here:

**Direct download:** https://drive.google.com/uc?export=download&id=1nNNt4hMk5O-czmStw9_6LK-CqRCVh9RF

**Or via browser:** https://drive.google.com/file/d/1nNNt4hMk5O-czmStw9_6LK-CqRCVh9RF/view?usp=sharing

Or download programmatically:
```bash
pip install gdown
gdown --id 1nNNt4hMk5O-czmStw9_6LK-CqRCVh9RF -O model/best.pt
```

Place the downloaded file at `model/best.pt`.

**4. Run the API:**
```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs (Swagger UI) at `http://127.0.0.1:8000/docs`.

---

## API Usage

### `GET /health`
Check server and model status.

**Response:**
```json
{
  "status": "ok",
  "model_path": "model/best.pt",
  "model_loaded": false,
  "model_available": true
}
```

### `POST /detect`
Accepts an image file, returns detections + safety assessment.

**Request (multipart/form-data):**
```
file: <image.jpg>
```

**Example (curl):**
```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@sample.jpg"
```

**Response:**
```json
{
  "filename": "sample.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "Hardhat",
      "confidence": 0.7831,
      "bbox": [647.95, 83.42, 870.79, 233.42]
    },
    {
      "class_id": 1,
      "class_name": "NO-Hardhat",
      "confidence": 0.4071,
      "bbox": [964.29, 121.39, 1146.9, 242.82]
    }
  ],
  "safety": {
    "status": "VIOLATION",
    "risk_level": "LOW",
    "message": "Potential PPE violation detected. 1 NO-Hardhat detection(s) were identified.",
    "hardhat_count": 1,
    "no_hardhat_count": 1,
    "compliance_rate": 0.5,
    "high_confidence_hardhat": 1,
    "high_confidence_no_hardhat": 0,
    "uncertain_detections": 1,
    "requires_review": true
  }
}
```

### `POST /ask`
Accepts an image + a natural-language question. Answers using a hand-written intent-routing + reasoning + confidence-guardrail pipeline (see memo for full design explanation).

**Request (multipart/form-data):**
```
question: "Is anyone not wearing a hardhat?"
file: <image.jpg>
```

**Example (curl):**
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -F "question=Is anyone not wearing a hardhat?" \
  -F "file=@sample.jpg"
```

**Response (on-topic question, detector called):**
```json
{
  "question": "Is anyone not wearing a hardhat?",
  "answer": "I detected 1 NO-Hardhat instance(s).",
  "used_detector": true,
  "detections": [ ... ],
  "safety": { ... }
}
```

**Response (off-topic question, detector NOT called):**
```json
{
  "question": "What's the capital of France?",
  "answer": "This question does not appear to relate to the uploaded image's PPE or safety content, so no detection was performed.",
  "used_detector": false,
  "detections": null,
  "safety": null
}
```

---

## Running tests
```bash
python -m pytest tests/ -v
```
Covers: API health/validation, reasoning-layer logic across compliant/violation/uncertain/no-detection scenarios, and model loading + inference sanity checks.

---

## Reproducing the training run

**1. Dataset:**
Sourced from [Hard Hats Dataset, Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5) (CC BY 4.0), version 1, YOLOv8 export format. 19,745 images total (13,782 train / 3,962 valid / 2,001 test), classes: `Hardhat`, `NO-Hardhat`.

Download via Roboflow (requires a free API key):
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("roboflow-universe-projects").project("hard-hats-fhbh5")
version = project.version(1)
dataset = version.download("yolov8")
```

**2. Training environment:**
- Google Colab free tier, NVIDIA T4 GPU (14.9GB VRAM)
- Ultralytics 8.4.149, RT-DETR-L (`rtdetr-l.pt` pretrained checkpoint)

**3. Training command:**
```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
model.train(
    data="path/to/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    save_period=5,
    project="runs",
    name="final_run"
)
```

**4. Hyperparameters:** batch=8, imgsz=640, optimizer=auto, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, patience=7, amp=True. All other values left at Ultralytics defaults.

**5. Training time & outcome:** Training was run across two Colab sessions due to free-tier compute limits (~16-17 min/epoch on T4). It was halted at epoch 22 after loss values became `NaN` — likely caused by an optimizer-state discontinuity from resuming training across sessions. The final model uses the checkpoint from **epoch 21** (`best.pt`, selected automatically by Ultralytics based on validation fitness), with test-set mAP50 = 0.921, mAP50-95 = 0.569. Full details and honest discussion of this in the accompanying memo (Section 3 & 6).

**6. Resuming training from a checkpoint** (if you hit a compute limit yourself):
```python
from ultralytics import RTDETR

model = RTDETR("path/to/weights/last.pt")
model.train(resume=True, save_period=5)
```
Note: `args.yaml` in the run folder must have a `data:` path that resolves correctly in whatever environment you're resuming in — update it manually if resuming on a different machine/account.

---

## Known limitations

See the accompanying memo for full failure-case analysis. Briefly:
- Occasional false positives on hat-like or rounded objects that aren't actual hardhats
- Occasional hallucinated detections in low-light images
- Detects hardhat presence, not whether it's actually being worn (vs. carried)
- Low-confidence misclassifications near the guardrail threshold can be silently discarded rather than flagged for review
- Occasional duplicate/overlapping detections on closely-spaced people