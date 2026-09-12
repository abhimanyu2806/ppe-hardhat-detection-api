from app.reasoning import analyze_safety


def test_compliant_scene():
    detections = [
        {
            "class_name": "Hardhat",
            "confidence": 0.92,
        },
        {
            "class_name": "Hardhat",
            "confidence": 0.87,
        },
    ]

    result = analyze_safety(detections)

    assert result["status"] == "COMPLIANT"
    assert result["hardhat_count"] == 2
    assert result["no_hardhat_count"] == 0
    assert result["compliance_rate"] == 1.0
    assert result["risk_level"] == "LOW"


def test_ppe_violation():
    detections = [
        {
            "class_name": "Hardhat",
            "confidence": 0.91,
        },
        {
            "class_name": "NO-Hardhat",
            "confidence": 0.88,
        },
    ]

    result = analyze_safety(detections)

    assert result["status"] == "VIOLATION"
    assert result["hardhat_count"] == 1
    assert result["no_hardhat_count"] == 1
    assert result["compliance_rate"] == 0.5
    assert result["risk_level"] == "MEDIUM"


def test_high_risk_scene():
    detections = [
        {
            "class_name": "NO-Hardhat",
            "confidence": 0.91,
        },
        {
            "class_name": "NO-Hardhat",
            "confidence": 0.86,
        },
        {
            "class_name": "Hardhat",
            "confidence": 0.90,
        },
    ]

    result = analyze_safety(detections)

    assert result["status"] == "VIOLATION"
    assert result["no_hardhat_count"] == 2
    assert result["risk_level"] == "HIGH"


def test_uncertain_detection():
    detections = [
        {
            "class_name": "Hardhat",
            "confidence": 0.50,
        }
    ]

    result = analyze_safety(detections)

    assert result["status"] == "COMPLIANT"
    assert result["uncertain_detections"] == 1
    assert result["requires_review"] is True


def test_no_usable_detections():
    detections = [
        {
            "class_name": "Hardhat",
            "confidence": 0.20,
        }
    ]

    result = analyze_safety(detections)

    assert result["status"] == "NO_DETECTIONS"
    assert result["risk_level"] == "UNKNOWN"
    assert result["compliance_rate"] is None