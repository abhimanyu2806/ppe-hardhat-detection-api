from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.detector import PPEDetector
from app.reasoning import analyze_safety


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="PPE Safety Detection API",
    description="RT-DETR based hard-hat detection and safety reasoning API",
    version="1.0.0"
)


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

# Build the model path relative to the project directory.
# This means the API can be started from different working
# directories without breaking the model path.

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"

# Model is loaded lazily when first needed.
detector = None


# --------------------------------------------------
# Intent routing
# --------------------------------------------------

# These are terms strongly associated with PPE/image questions.
# This is deliberately a simple, deterministic rule-based
# router so the system remains easy to explain and reproduce.

PPE_KEYWORDS = [
    "hardhat",
    "hard hat",
    "helmet",
    "ppe",
    "no-hardhat",
    "no hardhat",
    "without hardhat",
    "without a hardhat",
    "not wearing",
    "wearing a hardhat",
    "safety helmet",
    "ppe violation",
    "ppe compliance",
    "ppe risk"
]

IMAGE_CONTEXT_KEYWORDS = [
    "image",
    "picture",
    "photo",
    "worker",
    "workers",
    "wearing",
    "violation",
    "compliance",
    "compliant",
    "risk",
    "detect",
    "detection",
    "how many",
    "count",
    "identify"
]


def is_image_related_question(question: str) -> bool:
    """
    Decide whether the question requires the PPE detector.

    PPE-specific terms trigger detection directly.

    Some more general image terms also trigger detection because
    they indicate that the user is asking about the uploaded image.
    """

    question_lower = question.lower()

    # Strong PPE-specific match
    if any(keyword in question_lower for keyword in PPE_KEYWORDS):
        return True

    # More general image/PPE-context match
    if any(keyword in question_lower for keyword in IMAGE_CONTEXT_KEYWORDS):
        return True

    return False


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_detector():
    """
    Load the RT-DETR model only when it is first needed.

    The loaded model is reused for later requests.
    """

    global detector

    if detector is None:

        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model not found at {MODEL_PATH}. "
                    "Copy best.pt into the project model folder."
                )
            )

        detector = PPEDetector(
            model_path=str(MODEL_PATH),
            confidence_threshold=0.25
        )

    return detector


def load_image_from_upload(image_bytes: bytes) -> Image.Image:
    """
    Validate and convert uploaded image bytes into an RGB PIL image.
    """

    try:
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )


def validate_image_file(file: UploadFile):
    """
    Check that the uploaded file is an image.
    """

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "PPE Safety Detection API is running.",
        "endpoints": [
            "/detect",
            "/ask",
            "/health"
        ]
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_loaded": detector is not None,
        "model_available": MODEL_PATH.exists()
    }


# --------------------------------------------------
# Detection endpoint
# Part A
# --------------------------------------------------

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accept an image and run RT-DETR inference.

    Returns:
        - detected objects
        - class names
        - confidence scores
        - bounding boxes
        - PPE safety assessment
    """

    validate_image_file(file)

    model = get_detector()

    image_data = await file.read()

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    image = load_image_from_upload(image_data)

    detections = model.detect(image)
    safety = analyze_safety(detections)

    return {
        "filename": file.filename,
        "detections": detections,
        "safety": safety
    }


# --------------------------------------------------
# Question / reasoning endpoint
# Part B
# --------------------------------------------------

@app.post("/ask")
async def ask(
    question: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Accept an image and a natural-language question.

    Processing pipeline:

    1. Intent routing
       Decide whether the question is related to the
       uploaded image/PPE.

    2. Detection
       If relevant, run RT-DETR on the image.

    3. Structured reasoning
       Convert detections into a safety assessment.

    4. Confidence guardrail
       Do not guess when there is insufficient evidence.

    5. Answer generation
       Produce a deterministic answer from the structured
       detection and reasoning results.

    No agentic framework or external LLM is used.
    """

    question_clean = question.strip()

    if not question_clean:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------
    # Step 1: Intent routing
    # --------------------------------------------------

    if not is_image_related_question(question_clean):

        return {
            "question": question_clean,
            "answer": (
                "This question does not appear to relate to "
                "the uploaded image's PPE or safety content, "
                "so no detection was performed."
            ),
            "used_detector": False,
            "detections": None,
            "safety": None
        }

    # --------------------------------------------------
    # Step 2: Validate image and run detector
    # --------------------------------------------------

    validate_image_file(file)

    model = get_detector()

    image_data = await file.read()

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    image = load_image_from_upload(image_data)

    detections = model.detect(image)

    # --------------------------------------------------
    # Step 3: Structured safety reasoning
    # --------------------------------------------------

    safety = analyze_safety(detections)

    # --------------------------------------------------
    # Step 4: Confidence guardrail
    # --------------------------------------------------

    if safety["status"] == "NO_DETECTIONS":

        return {
            "question": question_clean,
            "answer": (
                "Insufficient information to answer confidently. "
                "No sufficiently confident PPE-related detections "
                "were identified in this image."
            ),
            "used_detector": True,
            "detections": detections,
            "safety": safety
        }

    # --------------------------------------------------
    # Step 5: Generate deterministic answer
    # --------------------------------------------------

    answer = build_answer(
        question_clean,
        safety
    )

    return {
        "question": question_clean,
        "answer": answer,
        "used_detector": True,
        "detections": detections,
        "safety": safety
    }


# --------------------------------------------------
# Structured answer generation
# --------------------------------------------------

def build_answer(question: str, safety: dict) -> str:
    """
    Convert the structured safety result into a natural-language
    answer using deterministic rules.

    No LLM or agent framework is used.
    """

    question_lower = question.lower()

    # --------------------------------------------------
    # Hardhat count
    # --------------------------------------------------

    asks_for_count = (
        "how many" in question_lower
        or "count" in question_lower
    )

    asks_for_hardhat = (
        "hardhat" in question_lower
        or "hard hat" in question_lower
        or "helmet" in question_lower
    )

    asks_for_no_hardhat = (
        "no-hardhat" in question_lower
        or "no hardhat" in question_lower
        or "without hardhat" in question_lower
        or "without a hardhat" in question_lower
        or "not wearing" in question_lower
    )

    if asks_for_count and asks_for_hardhat and not asks_for_no_hardhat:

        return (
            f"There are {safety['hardhat_count']} "
            "Hardhat detection(s)."
        )

    # --------------------------------------------------
    # NO-Hardhat count
    # --------------------------------------------------

    if asks_for_count and asks_for_no_hardhat:

        return (
            f"There are {safety['no_hardhat_count']} "
            "NO-Hardhat detection(s)."
        )

    # --------------------------------------------------
    # Compliance
    # --------------------------------------------------

    if (
        "compliance" in question_lower
        or "compliant" in question_lower
    ):

        if safety["compliance_rate"] is None:

            return (
                "Compliance cannot be determined because "
                "there are not enough sufficiently confident "
                "PPE detections."
            )

        percentage = safety["compliance_rate"] * 100

        return (
            f"The detected PPE compliance rate is "
            f"{percentage:.1f}%."
        )

    # --------------------------------------------------
    # Violation / safety
    # --------------------------------------------------

    if (
        "violation" in question_lower
        or "unsafe" in question_lower
        or "is it safe" in question_lower
        or "is this safe" in question_lower
    ):

        if safety["status"] == "VIOLATION":

            return (
                "A potential PPE violation was detected. "
                f"{safety['no_hardhat_count']} NO-Hardhat "
                "detection(s) were identified."
            )

        if safety["status"] == "COMPLIANT":

            return (
                "No NO-Hardhat detections were identified "
                "among the sufficiently confident detections."
            )

    # --------------------------------------------------
    # Risk level
    # --------------------------------------------------

    if "risk" in question_lower:

        risk = safety["risk_level"]

        if risk == "HIGH":

            return (
                "The assessed PPE risk level is HIGH because "
                "multiple high-confidence NO-Hardhat detections "
                "were identified."
            )

        if risk == "MEDIUM":

            return (
                "The assessed PPE risk level is MEDIUM because "
                "one high-confidence NO-Hardhat detection "
                "was identified."
            )

        if safety["no_hardhat_count"] > 0:

            return (
                "The assessed PPE risk level is LOW, although "
                "a lower-confidence NO-Hardhat detection was "
                "identified."
            )

        return (
            "The assessed PPE risk level is LOW because no "
            "NO-Hardhat detections were identified."
        )

    # --------------------------------------------------
    # Why / explanation
    # --------------------------------------------------

    if "why" in question_lower:

        return safety["message"]

    # --------------------------------------------------
    # General hardhat question
    # --------------------------------------------------

    if asks_for_hardhat and not asks_for_no_hardhat:

        return (
            f"I detected {safety['hardhat_count']} Hardhat "
            "instance(s)."
        )

    # --------------------------------------------------
    # General NO-Hardhat question
    # --------------------------------------------------

    if asks_for_no_hardhat:

        return (
            f"I detected {safety['no_hardhat_count']} "
            "NO-Hardhat instance(s)."
        )

    # --------------------------------------------------
    # Default fallback
    # --------------------------------------------------

    return safety["message"]