# PPE Safety Detection API

RT-DETR based hard-hat detection with a lightweight rule-based reasoning layer for answering natural-language questions about PPE compliance in images.

Built for: Pre-Hackathon Screening, Round 1 — Constrained Object Detection & Reasoning API.

---

## Live deployment

> **Live URL:** https://ppe-hardhat-detection-api.onrender.com
> **Interactive docs:** https://ppe-hardhat-detection-api.onrender.com/docs
> **Health check:** https://ppe-hardhat-detection-api.onrender.com/health
>
> To use any endpoint, open the `/docs` link above, expand an endpoint, click **"Try it out"**, then **"Execute"**.
>
> ⚠️ **Known limitation:** This is deployed on Render's **free tier**, which provides only 512MB RAM — tight for RT-DETR-L + PyTorch. `/detect` and `/ask` may intermittently return a `502` error under this memory constraint, even though the container starts correctly and `/health` responds normally. **This is a hosting resource limitation, not a code defect** — the exact same requests succeed reliably when run locally or via Docker with adequate memory (see "Running with Docker" below). This tradeoff was accepted given the project timeline; a paid tier or a smaller model would resolve it in production.

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
│   ├── main.py            # FastAPI app, both endpoints, logging
│   ├── detector.py         # RT-DETR wrapper
│   └── reasoning.py        # Rule-based safety analysis
├── model/
│   └── best.pt              # Trained checkpoint (download separately, see below)
├── tests/
│   ├── test_api.py
│   ├── test_reasoning.py
│   ├── test_model.py
│   └── image.jpg            # Sample image used by test_model.py
├── train.py                 # Training script (fine-tune RT-DETR)
├── eval.py                  # Evaluation script (test-set metrics)
├── Dockerfile                # Container build (see "Running with Docker" below)
├── entrypoint.sh             # Container startup script (auto-downloads model if missing)
├── .dockerignore
├── requirements.txt
└── README.md
```

**Note:** `model/best.pt` is not included in this repo (GitHub's 100MB file limit) — see "Get the model weights" below. Similarly, no dataset images are included — see "Reproducing training and evaluation" below for the download command.

---

## Setup

**1. Clone the repo and create a virtual environment:**
```bash
git clone https://github.com/abhimanyu2806/ppe-hardhat-detection-api.git
cd ppe-hardhat-detection-api
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

Tested with Python 3.11.

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
gdown "https://drive.google.com/uc?id=1nNNt4hMk5O-czmStw9_6LK-CqRCVh9RF" -O model/best.pt
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

**Response (insufficient information — detector called, but no usable detections):**
```json
{
  "question": "Is anyone not wearing a hardhat?",
  "answer": "Insufficient information to answer confidently -- no sufficiently confident PPE-related detections were identified in this image.",
  "used_detector": true,
  "detections": [],
  "safety": {
    "status": "NO_DETECTIONS",
    "risk_level": "UNKNOWN",
    "message": "No sufficiently confident PPE-related detections were identified.",
    "hardhat_count": 0,
    "no_hardhat_count": 0,
    "compliance_rate": null,
    "high_confidence_hardhat": 0,
    "high_confidence_no_hardhat": 0,
    "uncertain_detections": 0,
    "requires_review": false
  }
}
```
This triggers when an on-topic question is asked but the image contains no people/hardhats, or all detections fall below the confidence floor — the system explicitly declines to guess rather than returning a potentially wrong answer.

---

## Running with Docker

**1. Build the image:**
```bash
docker build -t ppe-detection-api .
```

**2. Run it, mounting your local model folder** (so weights don't need to be baked into the image):
```bash
docker run -p 8000:8000 -v /path/to/your/model:/app/model ppe-detection-api
```

**Or, without a local model folder** — the container's `entrypoint.sh` will automatically download `best.pt` from Drive on startup if it's not already present at `/app/model/best.pt`. This is how the live Render deployment works (no manual file copy needed).

**3. Access it exactly like the local (non-Docker) setup:**
```
http://localhost:8000/docs
```

Note: use `localhost`, not the `0.0.0.0` address Uvicorn prints in its startup log — that's a bind address, not a browser URL.

---

## Logging

The API logs to both the console and a persistent `app.log` file (excluded from git). Logged events include: startup + model path check, every `/detect` and `/ask` call (filename/question), the intent-routing decision for each `/ask` call, detection results, confidence-guardrail triggers, and errors (missing model, invalid images, inference failures).

---

## Error handling

Beyond standard input validation (non-image files, corrupt images, empty uploads, empty questions), both `/detect` and `/ask` explicitly catch and log model inference failures, returning a clean `500` response rather than an unhandled crash. Known, accepted gaps (not fixed given the project timeline): no explicit request size limit for very large file uploads.

---

## Running tests
```bash
python -m pytest tests/ -v
```
Covers: API health/validation, reasoning-layer logic across compliant/violation/uncertain/no-detection scenarios, and model loading + inference sanity checks.

---

## Reproducing training and evaluation

**1. Dataset:**
Sourced from [Hard Hats Dataset, Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5) (CC BY 4.0), version 1, YOLOv8 export format. 19,745 images total (13,782 train / 3,962 valid / 2,001 test), classes: `Hardhat`, `NO-Hardhat`.

**2. Training environment used for the submitted model:**
- Google Colab free tier, NVIDIA T4 GPU (14.9GB VRAM)
- Ultralytics 8.4.149, RT-DETR-L (`rtdetr-l.pt` pretrained checkpoint)
- Hyperparameters: batch=8, imgsz=640, optimizer=auto, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, patience=7, amp=True. All other values left at Ultralytics defaults.

**3. Training time & outcome:** Training was run across two Colab sessions due to free-tier compute limits (~16-17 min/epoch on T4). It was halted at epoch 22 after loss values became `NaN` — likely caused by an optimizer-state discontinuity from resuming training across sessions. The final model uses the checkpoint from **epoch 21** (`best.pt`, selected automatically by Ultralytics based on validation fitness), with test-set mAP50 = 0.921, mAP50-95 = 0.569. Full details and honest discussion of this in the accompanying memo (Section 3 & 6).

**4. `train.py` and `eval.py` require an actual `data.yaml` pointing to real, downloaded images and labels on your machine** — they cannot run against just the dataset's config file alone, since that only contains paths, not the images themselves. To run either script yourself:

**Download the dataset** (requires a free Roboflow account/API key):
```bash
pip install roboflow
python -c "from roboflow import Roboflow; rf = Roboflow(api_key='YOUR_KEY'); p = rf.workspace('roboflow-universe-projects').project('hard-hats-fhbh5'); p.version(1).download('yolov8')"
```
This creates a folder (e.g. `Hard-Hats-1/`) containing `train/`, `valid/`, `test/` subfolders and a `data.yaml`.

**Note on paths:** Roboflow's exported `data.yaml` uses relative paths (`../train/images`) that resolve differently depending on where you run commands from. If you hit path errors, either run `train.py`/`eval.py` from inside the downloaded dataset folder, or edit `data.yaml` to use absolute paths for your machine, e.g.:
```yaml
train: /full/path/to/Hard-Hats-1/train/images
val: /full/path/to/Hard-Hats-1/valid/images
test: /full/path/to/Hard-Hats-1/test/images
nc: 2
names: ['Hardhat', 'NO-Hardhat']
```

**Train from scratch:**
```bash
python train.py --data /full/path/to/Hard-Hats-1/data.yaml
```

**Resume an interrupted run** (useful on free-tier compute with session limits):
```bash
python train.py --resume runs/detect/ppe_rtdetr_run/weights/last.pt
```

**Evaluate on the test split** (reproduces the exact metrics reported in the memo, Section 3):
```bash
python eval.py --weights model/best.pt --data /full/path/to/Hard-Hats-1/data.yaml
```

Both scripts check that the given paths exist and print a clear error message pointing back to this section if not, rather than a raw traceback.

---

## Known limitations

See the accompanying memo for full failure-case analysis. Briefly:
- Occasional false positives on hat-like or rounded objects that aren't actual hardhats
- Occasional hallucinated detections in low-light images
- Detects hardhat presence, not whether it's actually being worn (vs. carried)
- Low-confidence misclassifications near the guardrail threshold can be silently discarded rather than flagged for review
- Occasional duplicate/overlapping detections on closely-spaced people