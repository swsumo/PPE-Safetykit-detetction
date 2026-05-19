from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

# Class names exactly as stored in the trained model (index order matters)
CLASS_NAMES = [
    "Protective Helmet", "Shield", "Jacket", "Dust Mask",
    "Eye Wear", "Glove", "Protectuve Boots",
]

# BGR colors per class
COLORS = [
    (  0, 140, 255),  # Helmet   - orange
    ( 50, 205,  50),  # Shield   - green
    (255,  80,   0),  # Jacket   - blue
    (200,   0, 200),  # Dust Mask- purple
    (  0, 215, 255),  # Eye Wear - gold
    (  0,   0, 220),  # Glove    - red
    (255, 200,   0),  # Boots    - cyan
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
    def __init__(self, model_path, conf=0.5, iou=0.45):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou

    def predict(self, image_bgr):
        canvas, scale, pl, pt = _letterbox(image_bgr)

        # BGR → RGB, normalize, NHWC → NCHW
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        # Output shape: (1, 11, 8400) — 4 bbox + 7 classes
        raw = self.session.run(None, {self.input_name: blob})[0]
        preds = raw[0].T  # (8400, 11)

        class_ids = np.argmax(preds[:, 4:], axis=1)
        confs = preds[np.arange(len(class_ids)), 4 + class_ids]

        mask = confs >= self.conf
        preds, class_ids, confs = preds[mask], class_ids[mask], confs[mask]

        if len(preds) == 0:
            return image_bgr.copy(), set()

        # xywh in 640×640 space → xyxy in original image space
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = ((cx - bw / 2) - pl) / scale
        y1 = ((cy - bh / 2) - pt) / scale
        xywh = np.stack([x1, y1, bw / scale, bh / scale], axis=1).astype(int)

        idxs = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(), self.conf, self.iou)
        if len(idxs) == 0:
            return image_bgr.copy(), set()

        idxs = np.array(idxs).flatten()
        xywh, class_ids, confs = xywh[idxs], class_ids[idxs], confs[idxs]

        out = image_bgr.copy()
        detected = set()

        for (x, y, w, h), cid, cf in zip(xywh, class_ids, confs):
            cid = int(cid)
            color = COLORS[cid % len(COLORS)]
            label = f"{CLASS_NAMES[cid]} {cf:.2f}"
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(out, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            detected.add(CLASS_NAMES[cid].lower())

        return out, detected
