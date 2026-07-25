import datetime
import json
import logging
import os

import azure.functions as func
import numpy as np
import cv2

from ppe_detector import PPEDetector

app = func.FunctionApp()

# "protectuve boots" is a typo in the original dataset labels — kept intentionally
# so it matches the class name the model itself was trained on.
REQUIRED_PPE = {"protective helmet", "jacket", "glove", "protectuve boots"}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.onnx")

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = PPEDetector(MODEL_PATH)
    return _detector


@app.function_name(name="DetectPPE")
@app.blob_trigger(arg_name="inputblob", path="uploads/{name}", connection="AzureWebJobsStorage")
@app.blob_output(arg_name="outputblob", path="results/{name}.json", connection="AzureWebJobsStorage")
def detect_ppe(inputblob: func.InputStream, outputblob: func.Out[str]):
    logging.info(f"Processing blob: {inputblob.name} ({inputblob.length} bytes)")

    img_array = np.frombuffer(inputblob.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        outputblob.set(json.dumps({
            "source_blob": inputblob.name,
            "error": "could not decode image",
        }))
        return

    detections = get_detector().predict(image)
    detected_classes = sorted({d["class"] for d in detections})
    missing = sorted(REQUIRED_PPE - set(detected_classes))

    result = {
        "source_blob": inputblob.name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "detections": detections,
        "detected_classes": detected_classes,
        "missing": missing,
        "compliant": len(missing) == 0,
    }

    outputblob.set(json.dumps(result))
    logging.info(f"Result written for {inputblob.name}: compliant={result['compliant']}")
