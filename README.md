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

**Note:** `model/best.pt` is not included in this repo (GitHub's 100MB file limit) — see "Get the model weights" below. Similarly, no dataset images are included — see "Dataset & reproducing training" below for the download command.

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

Open Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

## Dataset & reproducing training

### Dataset

**Name:** Hard Hats Dataset
**Link:** https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5
**License:** CC BY 4.0
**Size:** 19,745 images (13,782 train / 3,962 valid / 2,001 test)
**Classes:** `Hardhat`, `NO-Hardhat`

### How to download it (step-by-step, for anyone unfamiliar with Roboflow)

**1. Create a free Roboflow account**
Go to https://roboflow.com and sign up (free, no credit card needed).

**2. Get your API key**
After signing in, go to https://app.roboflow.com/settings/api (or: click your profile icon top-right → **Settings** → **API Keys** in the left sidebar). Copy the key shown there — it's a long string of letters/numbers, e.g. `abc123XYZ...`.

**3. Install the Roboflow Python package**
```bash
pip install roboflow
```

**4. Run this command to download the dataset**, replacing `YOUR_KEY` with the API key you copied in step 2:
```bash
python -c "from roboflow import Roboflow; rf = Roboflow(api_key='YOUR_KEY'); p = rf.workspace('roboflow-universe-projects').project('hard-hats-fhbh5'); p.version(1).download('yolov8')"
```
**What this does, in plain terms:** it logs into your Roboflow account using your key, finds this specific public dataset, and downloads all 19,745 images plus their labels onto your computer, into a new folder (something like `Hard-Hats-1/`) created in whatever directory you ran the command from.

**5. Confirm it worked**
Look for the new folder — it should contain three subfolders (`train/`, `valid/`, `test/`, each with an `images/` and `labels/` folder inside) and one file called `data.yaml`. If you see these, the download succeeded.

**Note on paths:** the `data.yaml` file uses relative paths (`../train/images`) that only work correctly if you run `train.py`/`eval.py` from certain locations. If you get a "file not found" error later, either run those scripts from inside the downloaded dataset folder, or open `data.yaml` and replace the paths with the full, absolute path to each folder on your machine, e.g.:
```yaml
train: /full/path/to/Hard-Hats-1/train/images
val: /full/path/to/Hard-Hats-1/valid/images
test: /full/path/to/Hard-Hats-1/test/images
nc: 2
names: ['Hardhat', 'NO-Hardhat']
```

### Training environment used for the submitted model
- Google Colab free tier, NVIDIA T4 GPU (14.9GB VRAM)
- Ultralytics 8.4.149, RT-DETR-L (`rtdetr-l.pt` pretrained checkpoint)
- Hyperparameters: batch=8, imgsz=640, optimizer=auto, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, patience=7, amp=True. All other values left at Ultralytics defaults.

### Training time & outcome
Training was run across two Colab sessions due to free-tier compute limits (~16-17 min/epoch on T4). It was halted at epoch 22 after loss values became `NaN` — likely caused by an optimizer-state discontinuity from resuming training across sessions. The final model uses the checkpoint from **epoch 21** (`best.pt`, selected automatically by Ultralytics based on validation fitness), with test-set mAP50 = 0.921, mAP50-95 = 0.569. Full details and honest discussion of this in the accompanying memo.

### Train from scratch (once you've downloaded the dataset above)
```bash
python train.py --data /full/path/to/Hard-Hats-1/data.yaml
```

### Resume an interrupted run (useful on free-tier compute with session limits)
```bash
python train.py --resume runs/detect/ppe_rtdetr_run/weights/last.pt
```

### Evaluate on the test split (reproduces the exact metrics reported in the memo)
```bash
python eval.py --weights model/best.pt --data /full/path/to/Hard-Hats-1/data.yaml
```

`train.py` and `eval.py` both require the actual downloaded dataset (not just `data.yaml` alone, since that only contains paths, not images) — and both check that given paths exist, printing a clear error message pointing back to this section if not, rather than a raw traceback.

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

## Reasoning Layer

The reasoning layer is implemented in:

```
app/main.py   (intent routing + answer generation)
app/reasoning.py   (structured safety analysis)
```

It handles question intents such as:

* `HARDHAT_COUNT` — "how many hardhats / people are wearing hardhats"
* `NO_HARDHAT_COUNT` — "is anyone not wearing a hardhat"
* `COMPLIANCE` — "what's the compliance rate"
* `VIOLATION` — "is this safe / is there a violation"
* `RISK` — "what's the risk level"
* `OFF_TOPIC` — unrelated to the image, detector is not called

The implementation is deterministic, hand-written rule-based logic. It does not use an external LLM, API key, or agentic framework.

---

## Guardrail Behavior

The system does not infer a PPE violation from detection counts alone. For example:

```
Hardhat detections < Person count
```

does not by itself mean a worker is missing a hardhat — the system only flags a violation when a `NO-Hardhat` detection is explicitly present, at or above the confidence floor.

For questions like:

```
Is anyone not wearing a hardhat?
```

the system looks specifically for `NO-Hardhat` detections, not just an absence of `Hardhat` detections. If there are zero usable detections after confidence filtering, the API returns:

```
"safety": { "status": "NO_DETECTIONS", ... }
"answer": "Insufficient information to answer confidently..."
```

instead of guessing. See "API Usage" above for the full example response.

---

## Running tests

```bash
python -m pytest tests/ -v
```

Current verified result:

```
10 passed
```

Covers: API health/validation, reasoning-layer logic across compliant/violation/uncertain/no-detection scenarios, and model loading + inference sanity checks.

---

## Known limitations

See the accompanying memo for full failure-case analysis. Briefly:
- Occasional false positives on hat-like or rounded objects that aren't actual hardhats
- Occasional hallucinated detections in low-light images
- Detects hardhat presence, not whether it's actually being worn (vs. carried)
- Low-confidence misclassifications near the guardrail threshold can be silently discarded rather than flagged for review
- Occasional duplicate/overlapping detections on closely-spaced people