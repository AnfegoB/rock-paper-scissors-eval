#%%
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from model import model #Cargamos el modelo


app = FastAPI()

opciones = ["Papel", "Piedra", "Tijera"] #Los indices deben coincidir con los establecidos en el .yaml en entrenamiento

umbral = 0.4

def predecir_gesto(imagen):
    """Esta funcion predice el gesto a partir de una imagen"""

    resultado = model(imagen)
    #Si ningun cuadro es encontrado:
    if len(resultado) == 0 or len(results[0].boxes) == 0:
        return None, 0.0
    
    #Extrayendo los resultados de las cajas generadas ouput: resultado[0].boxes
    boxes = resultado[0].boxes
    mejor_crit = 0
    prediction = None
    for box in boxes:
        crit = box.conf.cpu().item() #valor del criterio de confianza obtenido por el modelo
        indice = int(box.cls.cpu().item())
        if crit > mejor_crit and crit > umbral:
            prediction = indice
            mejor_crit = crit
    
    #Si no se cumple el umbral para el mejor caso:
    if mejor_crit is None:
        return None, 0.0
    
    return opciones[indice], mejor_crit

def vs_hands(player_A, player_B):
    if player_A == player_B:
        return "Empate", "Ambos jugadores escogieron la misma opcion"
    reglas = {'Tijera':'Papel', 'Piedra':'Tijera', 'Papel':'Piedra'} #Reglas como pares en diccionario

    if reglas[player_A] == player_B:
        return "Jugador A", f"{player_A} vence a {player_B}" #Ex: Si A es Tijera, llama a Papel, si este coincide con el que pierde con papel, gana
    else:
        return "Jugador B", f"{player_B} vence a {player_A}" #Ex: Caso contrario, si jugador escoge la opcion restante pierde A

imag = Image.open("test_im.jpg")
predecir_gesto(imag)

