from ultralytics import RTDETR
from PIL import Image


class PPEDetector:
    """
    RT-DETR based PPE detector.

    The model is fine-tuned for:
        0 -> Hardhat
        1 -> NO-Hardhat
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        self.model = RTDETR(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image: Image.Image):
        """
        Run RT-DETR inference on a PIL image.

        Returns a list of structured detections containing:
        - class ID
        - class name
        - confidence
        - bounding box
        """

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ]
                })

        return detections