Esta API permite evaluar 2 imagenes en las cuales se realiza un gesto de piedra, papel o tijera, 
a partir de un modelo de AI entrenado determina el ganador.

Para correr el Docker

Crear la imagen

$ docker build -t quipux_api .

Correr el contenedor

$ docker run -p 8000:8000 quipux_api

abrir para utilizar la API:

http://localhost:8000/docs

