from ultralytics import YOLO

#Empezando con un modelo de cero (no pre-entrenado)
model = YOLO("yolov8n.yaml")


model.train(data="data.yaml", epochs=300, device = "mps") #El modelo es entrenado en un Macbook con procesador M1
#El modelo se dejó correr solo por 100 epochs por motivos de tiempo

