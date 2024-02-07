# :paperclip: Clip detection with OpenCV & CNN

En este proyecto he puesto en práctica una limpieza y transformación de las imágenes iniciales para aislar los objetos a detectar posteriormente por la red convolucional, el proyecto sigue en proceso

### :pencil2: Prerequesitos

Para poder visualizar correctamente el proyecto deberás tener instaladas librerías de computer vision como *OpenCV*, y aquellas destinadas a trabajar con redes convolucionales como *tensorflow*. Por supuesto todas las habituales librerías de análisis y visualización.

```
pip install opencv-python
pip install tensorflow
```

### :put_litter_in_its_place: Limpieza

Para la limpieza de la imagen he empleado una serie de filtros y transformaciones que he encapsulado en una función y luego ha pasado una por una por todas las imágenes de los sets de entrenamiento y validación. Después de numerosas pruebas y errores, esta fue la fórmula mágica
Decidí dividir la imagen en la escala de color RGB, y tomando tan sólo la capa azul que era la que resultaba más fácil a la hora de retirar las imágenes horizontales que ensuciaban gran parte de la ilustración. 
Después, eliminé la que quedaba vertical y con mi imagen ya binarizada, aplicaba filtros y efectos que resaltaran mis objetos aislados, además de un conteo de contornos.
```
def cleaning_img(img): 
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    b,g,r = cv.split(img)
    blur = cv.GaussianBlur(b, (3,3), cv.BORDER_DEFAULT)
    ret, thresh = cv.threshold(blur, 225, 255, 1, cv.THRESH_BINARY)
    thresh = 255 - thresh
    canny = cv.Canny(thresh, 0, 25)
    dilated = cv.dilate(canny, (15,15), iterations=2)
    eroded = cv.erode(dilated, (7,7), iterations=1)
    #verticales
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 100))
    detected_lines_vertical = cv.morphologyEx(eroded, cv.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts_vertical, _ = cv.findContours(detected_lines_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    num_clips = len(cnts_vertical)
    mask_vertical = np.zeros_like(eroded)
    for cnt_vertical in cnts_vertical:
        x, y, w, h = cv.boundingRect(cnt_vertical)
        cv.rectangle(mask_vertical, (x, y), (x + w, y + h), 255, -1)
    mask_inverted_vertical = cv.bitwise_not(mask_vertical)
    clean_img = cv.bitwise_and(eroded, eroded, mask=mask_inverted_vertical)
    
    return clean_img, num_clips
    
```

### :construction_worker: Modelado

Tras la limpieza, puse en marcha un modelo tirando de Redes Convolucionales, los que empleamos para data no tabular/compleja (como imágenes). El modelo está en pruebas todavía, empecé por algo sencillo que, viniendo de una problemática de regresión, me permite seguir ajustándolo (una capa de salida que tira de una regresión lineal).

```
model = keras.Sequential([
    layers.Flatten(input_shape=(96, 96)), 
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='linear')    #capa de salida con activación linear para la problemática de regresión
])

#compilamos
model.compile(optimizer='adam',
              loss='mean_squared_error', 
              metrics=['mae'])        
```


## :wrench: Built With

* [Python](https://www.python.org/) - Lenguaje
* [OpenCV](https://opencv.org/) - Computer Vision
* [TensorFlow](https://www.tensorflow.org/api_docs) - CNN

## :star2: Authors

* **Lili Casanova** - 2024
