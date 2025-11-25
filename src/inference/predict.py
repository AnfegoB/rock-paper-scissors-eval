from ultralytics import YOLO

model = YOLO("best.pt")

#Esto permite revisar el funcionamiento del modelo a partir de la imagen captada por la camara del computador
results = model(source=0, show=True, conf=0.4, save=True, device="mps") #0.25 umbral