from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

# Class names exactly as stored in the trained model (index order matters)
CLASS_NAMES = [
    "Protective Helmet", "Shield", "Jacket", "Dust Mask",
    "Eye Wear", "Glove", "Protectuve Boots",
]


def _letterbox(image, size=640):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pt, pl = (size - nh) // 2, (size - nw) // 2
    canvas[pt:pt + nh, pl:pl + nw] = resized
    return canvas, scale, pl, pt


class PPEDetector:
    """Same detection math as the Flask app's detector.py, minus box drawing —
    the Function only needs structured results, not an annotated image."""

    def __init__(self, model_path, conf=0.5, iou=0.45):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou

    def predict(self, image_bgr):
        canvas, scale, pl, pt = _letterbox(image_bgr)

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        raw = self.session.run(None, {self.input_name: blob})[0]
        preds = raw[0].T  # (8400, 11)

        class_ids = np.argmax(preds[:, 4:], axis=1)
        confs = preds[np.arange(len(class_ids)), 4 + class_ids]

        mask = confs >= self.conf
        preds, class_ids, confs = preds[mask], class_ids[mask], confs[mask]

        if len(preds) == 0:
            return []

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = ((cx - bw / 2) - pl) / scale
        y1 = ((cy - bh / 2) - pt) / scale
        xywh = np.stack([x1, y1, bw / scale, bh / scale], axis=1).astype(int)

        idxs = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(), self.conf, self.iou)
        if len(idxs) == 0:
            return []

        idxs = np.array(idxs).flatten()
        xywh, class_ids, confs = xywh[idxs], class_ids[idxs], confs[idxs]

        detections = []
        for (x, y, w, h), cid, cf in zip(xywh, class_ids, confs):
            cid = int(cid)
            detections.append({
                "class": CLASS_NAMES[cid].lower(),
                "confidence": round(float(cf), 4),
                "box": [int(x), int(y), int(w), int(h)],
            })
        return detections
