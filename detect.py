from ultralytics import YOLO
model = YOLO('best.pt')

results  = model(source='demo4.mp4', show=True, conf=0.3, save=True)