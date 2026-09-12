def analyze_safety(detections):
    """
    Convert RT-DETR detections into a deterministic PPE
    safety assessment.

    This is a rule-based reasoning layer.
    It does not use agents or external LLM frameworks.
    """

    hardhat_count = 0
    no_hardhat_count = 0

    high_confidence_hardhat = 0
    high_confidence_no_hardhat = 0

    uncertain_detections = 0

    HIGH_CONFIDENCE_THRESHOLD = 0.70
    UNCERTAIN_THRESHOLD = 0.40

    for detection in detections:

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        if confidence < UNCERTAIN_THRESHOLD:
            continue

        if UNCERTAIN_THRESHOLD <= confidence < HIGH_CONFIDENCE_THRESHOLD:
            uncertain_detections += 1

        if class_name == "Hardhat":
            hardhat_count += 1

            if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                high_confidence_hardhat += 1

        elif class_name == "NO-Hardhat":
            no_hardhat_count += 1

            if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                high_confidence_no_hardhat += 1

    total = hardhat_count + no_hardhat_count

    # No usable detections
    if total == 0:
        return {
            "status": "NO_DETECTIONS",
            "risk_level": "UNKNOWN",
            "message": (
                "No sufficiently confident PPE-related "
                "detections were identified."
            ),
            "hardhat_count": 0,
            "no_hardhat_count": 0,
            "compliance_rate": None,
            "high_confidence_hardhat": 0,
            "high_confidence_no_hardhat": 0,
            "uncertain_detections": 0,
            "requires_review": False
        }

    compliance_rate = hardhat_count / total

    # Determine safety status
    if no_hardhat_count > 0:
        status = "VIOLATION"
    else:
        status = "COMPLIANT"

    # Determine risk level
    if high_confidence_no_hardhat >= 2:
        risk_level = "HIGH"

    elif high_confidence_no_hardhat == 1:
        risk_level = "MEDIUM"

    elif no_hardhat_count > 0:
        risk_level = "LOW"

    else:
        risk_level = "LOW"

    # Generate explanation
    if no_hardhat_count > 0:
        message = (
            f"Potential PPE violation detected. "
            f"{no_hardhat_count} NO-Hardhat detection(s) "
            f"were identified."
        )
    else:
        message = (
            "No NO-Hardhat detections were identified "
            "among the sufficiently confident detections."
        )

    # Flag scenes where uncertain detections may affect interpretation
    requires_review = (
        uncertain_detections > 0
        or high_confidence_no_hardhat > 0
    )

    return {
        "status": status,
        "risk_level": risk_level,
        "message": message,

        "hardhat_count": hardhat_count,
        "no_hardhat_count": no_hardhat_count,

        "compliance_rate": round(compliance_rate, 4),

        "high_confidence_hardhat": high_confidence_hardhat,
        "high_confidence_no_hardhat": high_confidence_no_hardhat,

        "uncertain_detections": uncertain_detections,

        "requires_review": requires_review
    }