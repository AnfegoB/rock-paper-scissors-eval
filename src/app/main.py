#%%
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from model import model #Cargamos el modelo
import numpy as np
import io
from utils import predecir_gesto, vs_hands

app = FastAPI()


@app.post("/play")
async def play(player_a : UploadFile =  File(...), player_b : UploadFile =  File(...) ):
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

