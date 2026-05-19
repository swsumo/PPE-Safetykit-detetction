"""
Run this once locally to export models/best.pt -> models/best.onnx
Requires ultralytics to be installed locally.
"""
from ultralytics import YOLO

model = YOLO("models/best.pt")
out = model.export(format="onnx", imgsz=640)
print(f"Exported: {out}")
