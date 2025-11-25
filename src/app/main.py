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
    if player_A == player_B:
        return "Tie", "Both players picked the same option"
    reglas = {'Tijera':'Papel', 'Piedra':'Tijera', 'Papel':'Piedra'} #Reglas como pares en diccionario

    if reglas[player_A] == player_B:
        return "Player A", f"{player_A} beats a {player_B}" #Ex: Si A es Tijera, llama a Papel, si este coincide con el que pierde con papel, gana
    else:
        return "Player B", f"{player_B} beats a {player_A}" #Ex: Caso contrario, si jugador escoge la opcion restante pierde A


@app.post("/play")
async def play(player_a : UploadFile = File(...), player_b : UploadFile = File(...) ):
    im_a = await player_a.read()
    im_b = await player_b.read()

    result_a, crit_a = predecir_gesto(im_a)
    result_b, crit_b = predecir_gesto(im_b)

    if result_a is None or result_b is None:
        return JSONResponse(status_code=400, content={"undecided": "No se encontró un gesto válido en alguna de las dos imagenes"})
    
    winner, reason = vs_hands(result_a,result_b)

    return{
           "player_a": {"prediction":result_a, "confidence":crit_a },
           "player_b": {"prediction":result_b, "confidence":crit_b },    
           "ganador": winner,
           "reason": reason
           }



#Rough testing
# imag = Image.open("test_im.jpg")
# imag2 = Image.open("test2_im.jpg")


# res1 = predecir_gesto(imag)[0]
# res2 = predecir_gesto(imag2)[0]


# print(vs_hands(res1,res2))