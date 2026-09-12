# test_model.py

from ultralytics import RTDETR


def test_model_loads_successfully():
    """
    Verifies the trained checkpoint loads correctly and
    exposes the expected PPE class names.
    """
    model = RTDETR("model/best.pt")

    assert model is not None
    assert set(model.names.values()) == {"Hardhat", "NO-Hardhat"}


def test_model_runs_inference_without_error():
    """
    Sanity check that the model can run a forward pass
    on a real image without throwing.
    """
    model = RTDETR("model/best.pt")
    results = model.predict(
        source="tests/image.jpg",
        conf=0.25,
        verbose=False
    )
    assert results is not None