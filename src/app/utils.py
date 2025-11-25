from ultralytics import YOLO
from PIL import Image
from model import model #Cargamos el modelo
import numpy as np
import io

opciones = ["Papel", "Piedra", "Tijera"] #Los indices deben coincidir con los establecidos en el .yaml en entrenamiento

umbral = 0.4

def predecir_gesto(imagen):
    """Esta funcion predice el gesto a partir de una imagen"""

    img =  Image.open(io.BytesIO(imagen)).convert("RGB")
    img_np = np.array(img)
    imp_pil = Image.fromarray(img_np) #https://github.com/ultralytics/ultralytics/issues/8172 un poco enredado al momento de transformar el input
    #Sin embargo, el modelo funciona mejor cuando el tipo de input es PIL y no un array
    resultado = model(imp_pil)
    print(resultado)
   
    #Si ningun cuadro es encontrado:
    if len(resultado) == 0 or len(resultado[0].boxes) == 0:
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
    "Esta funcion determina quien gana el juego"
    if player_A == player_B:
        return "Tie", "Both players picked the same option"
    reglas = {'Tijera':'Papel', 'Piedra':'Tijera', 'Papel':'Piedra'} #Reglas como pares en diccionario

    if reglas[player_A] == player_B:
        return "Player A", f"{player_A} beats a {player_B}" #Ex: Si A es Tijera, llama a Papel, si este coincide con el que pierde con papel, gana
    else:
        return "Player B", f"{player_B} beats a {player_A}" #Ex: Caso contrario, si jugador escoge la opcion restante pierde A
