import cv2  ## esta libreria nos sirve para poder abrir la camara, dibujar lineas, detectar teclas
import mediapipe as mp
import numpy as np  ## en Numpy, vamos a trabajar con matricez, este nos va a servir para poder crear nuestra pizarron digital
from mediapipe import solutions  ## esta libreria nos ayuda a detectar manos, rostro o gestos e incluso nuestros deditos

##ahora vamos a traer algunas herramientas de "madiapipe", entonces lo vamos a guardar en las variables:

##"hands": que guarda al metodo/herramienta "solutions.hands", este contiene el detector de manos, los "landmark" que nos ayudan a saber
## donde esta mi mano, mis dedos y como voy haciendo los movimientos

##"drawing_utils": que guarda al metodo/herremientas "solutions.drawing_utils"
# sirve par dibujar los puntos de las manos y las lineas para unir esos puntos (solutions.drawing_utils)

hands = solutions.hands
drawing_utils = solutions.drawing_utils


class AirPainter:  ## aqui estamos creando una clase
    ## tenemos el constructor inicial, donde pasamos como parametro el ancho y largo que queremos para nuestra pantalla
    def __init__(self, width=1280, height=720):
        self.width = width  ## guardamos el ancho y alto de esta instancia
        self.height = height  ## aquí solo tenemos los atributos de la instancia

        # Configurar MediaPipe
        ## estamos creando un atributo de clase ".mp_hands",
        # que nos ayuda a traer a "hands" que contiene las herremientas que antes mencione, linea 14 y linea 8
        self.mp_hands = hands
        self.hands = self.mp_hands.Hands(  ## creamos un objeto
            static_image_mode=False,  # aqui estamos indicando que es un video en tiempo real y no imagenes fijas
            max_num_hands=1,  # aqui solo detectamos una mano
            min_detection_confidence=0.7,
            ##Se usa para la primera detección,"MediaPipe" busca una mano en el frame y solo
            # la considera válida si tiene al menos 70% de confianza de que ES UNA MANOOO
            min_tracking_confidence=0.7)  ## Se usa para hacer el  seguimiento después de la 1era detectada, entonces una vez que
        # ya encontró la mano,en los frames siguientes intenta seguirla, en dado caso  Si la confianza del seguimiento baja del 70%, entonces vuelve a hacer
        # una detección completa (más costosa).

        self.mp_drawing = drawing_utils  ## aqui pasa lo mismo que en la linea 26
        ##estamos creando instancia de clase , donde madamos a llamar a las herramientas que guarde en la linea 15
        ##te permite dibujar visualmente los puntos de la mano en la imagen

        ###################################################################
        ## CONFIGURACIÓN DEL LIENZO (DONDE SE DIBUJA)
        ###################################################################
        ## con "np.zeros" preparamos el "lienzo digital" en la memoria... aun no has dibujado nada
        ## pasamos como parametro de "np.zeros" las dimensiones: alto, ancho y 3 canales de color
        ## el 3 nos indica que cada pixel tiene tres canales de color: Rojo(R), Verde(G) y Azul(B)
        ## la combinación de estos tres canales nos da cualquier color que queramos
        ## con "dtype=np.uint8" decimos que solo podemos representar colores del 0 al 255
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # <--- CORREGIDO: faltaba self.

        ## esta linea es necesaria para poder unir punto anterior con al punto actual
        self.prev_point = None  ## su proposito es guardar el punto/posisción anterior donde se coloca el dedo con el pincel, y se
        ## se inicializa en 0, porque al principio del programa, no tenemos ninguna posisción

        self.drawing = False  ## con ayuda de "drawing" detectamos si estas dibujando activamente o no, y se inicializa en "FALSE" porque al inicio
        ##no estamos dibujando, obvio que cuando dibujamos su valor cambia a "TRUE"

        self.color = (0, 255,
                      0)  ## Aquí establecemos el verde por defecto, si es "RGB" (por sus siglas en ingles ) la "G" corresponde a verde
        ## si en la posición de en medio pongo el 255, como valor maximo...quiere decir que es verde

        self.brush_size = 5  ## aqui estamos estableciendo el grosor de nuestro pincel

        ###################################################################
        ## PALETA DE COLORES DISPONIBLES
        ###################################################################
        ## creamos un diccionario que nos permite guardar todos los posibles colores
        ## con los que queramos pintar. La estructura es: 'nombre_del_color': (R, G, B)
        self.colors = {
            # clave  # valor ('R','G','B') -> ## establecemos los colores "RGB", según corresponda
            'rojo': (0, 0, 255),  # posición 0
            'verde': (0, 255, 0),  # 1
            'azul': (255, 0, 0),  # 2
            'amarillo': (0, 255, 255),  # 3
            'morado': (255, 0, 255),  # 4
            'cian': (255, 255, 0),  # 5
            'blanco': (255, 255, 255)  # 6
        }  ## Estoy haciendo colores "RGB"

        ####################################################################
        ## GESTIÓN DEL COLOR ACTUAL USANDO ÍNDICES
        ## ##################################################################
        ## esta linea extrae las claves (nombres de los colores) del diccionario
        ## y las convierte en una lista. Ejemplo: ['rojo', 'verde', 'azul', ...]
        ## esto nos permite acceder a los colores por su número de posición (índice)
        self.color_names = list(self.colors.keys())

        ## establecemos el índice del color actual en 1 (que corresponde al 'verde')
        ## los índices empiezan en 0: 0=rojo, 1=verde, 2=azul, etc.
        ## más adelante podemos cambiar este índice para cambiar de color fácilmente
        self.current_color_index = 1  ## este lo vamos a cambiar de color en la linea 376

    ## ##################################################################
    ## MÉTODO: DETECTAR QUÉ DEDOS ESTÁN LEVANTADOS
    ## ######################################################
    ## Analiza los 21 puntos de la mano que detecta MediaPipe y determina
    ## qué dedos están levantados (True) y cuáles están bajados (False)
    ##
    ## PARÁMETROS:
    ##   hand_landmarks: los puntos de la mano detectados por MediaPipe
    ##   handedness: 'Left' o 'Right' (qué mano es, izquierda o derecha)
    ##
    ## RETORNA:
    ##   una lista de 5 booleanos: [pulgar, índice, medio, anular, meñique]
    ##   True = dedo levantado, False = dedo bajado
    ## ###################################################################

    def detectar_gestos(self, hand_landmarks, handedness):
        """
        handedness: 'Left' o 'Right' (lo da MediaPipe)
        """
        dedos = []  ##Esta es la lista que nos va a ayudar a guardar los resultados
        #######################################
        # Detectar el pulgar según la mano
        # (tomando como referencia el efecto espejo de la cámara)
        #######################################

        ## A diferencia de los demás dedos, el pulgar no se mueve principalmente
        ## de arriba hacia abajo (eje Y), sino más hacia los lados (eje X).
        ## Por eso, para detectar si está levantado, analizamos su posición horizontal.

        ## Sabemos que:
        ## - 'Right' representa la mano derecha.
        ## - El landmark 4 corresponde a la punta del pulgar.
        ## - El landmark 3 corresponde a la articulación anterior del pulgar.

        ## Si nos colocamos desde la perspectiva de la cámara de la laptop
        ## (como si los "ojos" fueran la cámara),
        ## podemos notar que cuando el pulgar está levantado o extendido,
        ## la punta del dedo se desplaza más hacia la derecha.

        ## Entonces, si el landmark 4 (punta del pulgar)
        ## está más hacia la derecha respecto al landmark 3,
        ## consideramos que el pulgar está levantado.
        if handedness == 'Right':  ## mano (derecha)
            ## Mano derecha: pulgar levantado = punta (punto 4) está más a la derecha que la articulación (punto 3)
            ## es decir, X de la punta > X de la articulación
            pulgar_up = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
        else:  # Left - > mano (izquierda)
            # Mano izquierda: pulgar levantado = punta más a la izquierda
            ## es decir, X de la punta < X de la articulación
            pulgar_up = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x

        dedos.append(pulgar_up)  ## añadimos el resultado del pulgar a la lista

        #######################################################################################
        # Para los otros dedos (índice, medio, anular y meñique) la lógica funciona igual
        # para ambas manos, porque estamos comparando posiciones verticales en el eje Y.
        #######################################################################################

        ## IMPORTANTE: Las coordenadas en imágenes funcionan diferente al plano cartesiano.
        ##
        ## Normalmente en matemáticas pensamos:
        ##
        ## ↑ Y positivo
        ##
        ## Pero en visión por computadora/OpenCV las coordenadas comienzan desde la
        ## esquina superior izquierda de la pantalla:
        ##
        ## (0,0) ─────────→ X
        ##   |
        ##   |
        ##   ↓
        ##   Y
        ##
        ## Esto significa que:
        ##
        ## - Un valor Y más PEQUEÑO  -> está MÁS ARRIBA en la pantalla
        ## - Un valor Y más GRANDE   -> está MÁS ABAJO en la pantalla
        ##
        ## Por eso esta comparación:
        ##
        ## hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
        ##
        ## sí tiene sentido.
        ##
        ## Estamos diciendo:
        ## "La punta del dedo índice (landmark 8) está más arriba que su articulación
        ## (landmark 6)"
        ##
        ## Entonces:
        ##
        ## - Si la punta tiene una coordenada Y menor -> el dedo está levantado
        ## - Si la punta tiene una coordenada Y mayor -> el dedo está doblado o abajo
        ##
        ## Ejemplo:
        ##
        ## landmark 8 -> y = 100  (más arriba)
        ## landmark 6 -> y = 200  (más abajo)
        ##
        ## 100 < 200 -> TRUE
        ##
        ## Por lo tanto:
        ## el dedo índice está levantado.
        #######################################################################################

        indice_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
        dedos.append(indice_up)  ## aqui agregamos el resultado a nustra listita

        # Medio
        medio_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
        dedos.append(medio_up)  ## aqui agregamos el resultado a nustra listita

        # Anular
        anular_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
        dedos.append(anular_up)  ## aqui agregamos el resultado a nustra listita

        # Meñique
        menique_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
        dedos.append(menique_up)  ## aqui agregamos el resultado a nustra listita

        ##### como resultado final de esta función vamos a retornar la lista con todos los valores de los dedos
        return dedos

    ### okey y si ya se habia puesto dificil agarrate se pone peor
    #########################################################################################
    # La función de "get_finger_tip", sirve basicamente para poder hacer una conversión de cordenadas
    # es decir a cada landamarks de la mano le encontramos una cordenda, para la pantalla.... osea
    # vamos a convertir a los puntos  landamark a cordenadas que "OpenCV" pueda usar para dibujar en pantalla
    ##########################################################################################
    # Convierte las coordenadas normalizadas (0 a 1) de MediaPipe a coordenadas
    #  reales de píxeles (0 a ancho, 0 a alto) para poder dibujar en la imagen

    #  PARÁMETROS:
    #  hand_landmarks: los puntos de la mano detectados por MediaPipe
    #  finger_index: número del dedo (1=pulgar, 2=índice, 3=medio, 4=anular, 5=meñique)
    #
    #  RETORNA:
    #  una tupla (x, y) con las coordenadas en píxeles de la punta del dedo
    #
    ##########################################################################################

    def get_finger_tip(self, hand_landmarks, finger_index):
        """Obtiene las coordenadas de la punta de un dedo específico"""
        h, w = self.height, self.width  ## ## obtenemos las dimensiones de la pantalla
        tip_id = finger_index * 4  ## aquí es como si hicieramos un diccionario
        ## # finger_index=1 → 1*4 = 4  (pulgar)
        # finger_index=2 → 2*4 = 8  (índice)
        # finger_index=3 → 3*4 = 12 (medio)
        # finger_index=4 → 4*4 = 16 (anular)
        # finger_index=5 → 5*4 = 20 (meñique)

        ## convertimos coordenadas normalizadas (0 a 1) a píxeles (0 a ancho/alto) dependiendo del dedo que nos pidan
        ## digamos que "hand_landmarks.landmark[tip_id].x" nos trae un "0.5" como cordenada -> eso para la libreria de
        ##MediaPipe representa como la mitad de la pantalla, si lo mutiplicamos por "1280" osea el ancho, nos da el pixel
        ## en x de donde se encuntra muestro dedo
        x = int(hand_landmarks.landmark[tip_id].x * w)  # 0.5 * 1280 = 640
        ## digamos que "hand_landmarks.landmark[tip_id].y" nos trae un "0.3" como cordenada -> eso para la libreria de
        ##MediaPipe representa como la cuarta parte, si lo mutiplicamos por "720" osea el largo, nos da el pixel
        ## en y de donde se encuntra muestro dedo
        y = int(hand_landmarks.landmark[tip_id].y * h)  # 0.3 * 720  = 216
        ## en estas variables x & y vamos a tener guardada la cordenada real en pantalla de donde vamos dibujar
        return (x, y)  ## regresamos una cordenada real en donde esta apuntando nuestro dedo

    ## -------------------------------------------------------------------------
    ## MÉTODO: DIBUJAR LA PALETA DE COLORES EN PANTALLA
    ## -------------------------------------------------------------------------
    ## Dibuja una barra con todos los colores disponibles en la parte superior
    ## También marca con un borde blanco el color que está actualmente seleccionado
    ##
    ## PARÁMETROS:
    ##   frame: la imagen del video sobre la que vamos a dibujar la paleta
    ##
    ## RETORNA:
    ##   el frame con la paleta de colores dibujada encima
    ## -------------------------------------------------------------------------

    def draw_color_palette(self, frame):
        """Dibuja una paleta de colores en la parte superior"""
        palette_y = 100  ## posición vertical (Y) donde empieza la paleta (cambiado de 50 a 100)
        color_width = 60  ## ancho de cada cuadrito de color en píxeles
        start_x = 50  ## posición horizontal (X) donde empieza la paleta
        ## dentro del bucle for, traemos como parametro a:
        # "color_name" -> linea 85
        # "color_bgr" -> ya que a diferencia del ojo humano que utiliza el rgb, la libreria "OpenCV" utiliza bgr, y esto lo hacemos
        ##presisamente para poder hacer la coversión de color de un sistema a otro, es decir de "RGB" a -> "BGR"
        ## Luego vemos esta linea de aquí "in enumerate(self.colors.items()):" donde madamos a llamar al diccionario:
        ## el diccionario se encuentra en la linea -> 68 a 77... y con ".items()" indicamos que nos traiga información por pares
        # es decir tanto llave/clave con su valor/contenido
        for i, (color_name, color_bgr) in enumerate(
                self.colors.items()):  ## por cada color dentro del diccionario, vamos a :
            x = start_x + i * color_width
            # Vamos a calcular la posición horizontal (x) donde se dibujará cada rectángulo de color.
            # En cada iteración del ciclo, el valor iterado representa un color distinto.
            # Después multiplicamos ese valor por el ancho de cada cuadrito de color.
            #
            # Esto provoca que cada nuevo rectángulo se dibuje un poco más a la derecha del anterior,
            # formando así una fila horizontal de colores.

            ################ Ahora vamos a dibujar cada Cuadro de color
            # al metodo de "cv2.rectangle" le vamos a traer como paramtero:
            ##"frame" que es la imagen donde que remos dibujar, en este caso el frame de la camara
            ##"(x, palette_y)" -> la esquina superior izquierda
            ##"(x + color_width - 5" -> decimos que en la posición x, sumamos el ancho de cada cuadrito, menos 5 para dar espacio entre cada cuadrito
            cv2.rectangle(frame, (x, palette_y), (x + color_width - 5, palette_y + 40),
                          ## "palette_y + 40" -> posición vertical INFERIOR  ## "palette_y"-> en la posisción 50 arriba de la pantalla y le sumamos 40 pixeles
                          color_bgr, -1)  # el color que va a tener el rectangulo y con "-1" rellenar el rectangulo
            # Borde para el color seleccionado
            if i == self.current_color_index:  ## decimos que si el iterador es igual al color actual selecinado "self.current_color_index" linea 90, entonces vamos a:
                ##utilizar el metodo ".rectangle" y le pasamos como parametro el cuadrito que fuimos creando en  linea anterior
                cv2.rectangle(frame, (x, palette_y), (x + color_width - 5, palette_y + 40),
                              (255, 255, 255),
                              3)  ## y decimos que queremos que se le coloque un rectangulo blanco con grosor de 3
            # Texto del color
            ## escribimos el nombre del color (primeras 3 letras) dentro del rectángulo,en color blanco y con un tamaño de 0.4
            cv2.putText(frame, color_name[:3], (x + 5, palette_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # vamos a hacer nuestro Botón para limpiar
        ## entonces en la variable "clear_x", tenemos que:
        ## "start_x = 50" Primer color empieza en X=50
        ##"len(self.colors)" son 7 colores
        ##"* color_width" y que cada color ocupa 60 pixeles
        ## 7 cuadritos/colores entotal, por el tamaño de cada cuadrito 60 pixeles, nos da 420 pixeles
        ## si le sumamos una posición más al 420, es decir 50 pixeles más nos da como resultado 470 pixeles, que es en donde se dibujaria nuestro boton de borrar
        clear_x = start_x + len(self.colors) * color_width
        ## entonces dibujamos otro cuadrito
        cv2.rectangle(frame, (clear_x, palette_y), (clear_x + color_width - 5, palette_y + 40),
                      (128, 128, 128), -1)  ## de color gris y relleno del mismo color gris
        cv2.putText(frame, "LIMPIAR", (clear_x + 10, palette_y + 25),
                    ## decimos que queremos el texto de limippiar en este cuadrito gris
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  ## igual letras blancas, tamaño 0.4

        return frame  ## el resultado de esta función es devolvernos una pantalla/frame limpio y listo para dibujar

    ## -------------------------------------------------------------------------
    ## MÉTODO: VERIFICAR SELECCIÓN DE COLOR (VERSIÓN MEJORADA)
    ## -------------------------------------------------------------------------
    ## Versión mejorada que usa más puntos del dedo para verificar selección de color
    ## Ahora recibe hand_landmarks en lugar de x,y para poder verificar que el dedo
    ## esté bien extendido antes de seleccionar
    ##
    ## PARÁMETROS:
    ##   hand_landmarks: los puntos de la mano detectados por MediaPipe
    ##
    ## RETORNA:
    ##   True si seleccionó un color o limpió el canvas, False en caso contrario
    ## -------------------------------------------------------------------------

    def check_color_selection(self, hand_landmarks):
        """Versión mejorada que usa más puntos del dedo para verificar selección de color"""
        palette_y = 100  ## la paleta se comienza a dibujar a partir del pixel 100 en y (coherente con draw_color_palette)
        color_width = 60  ## que cada cuadrito de la paleta tendra de ancho 60 pixeles
        start_x = 50  ## la paleta se comienza a dibujar a partir del pixel 50 en x
        palette_height = 40  ## altura de la paleta

        # Aquí vamos a obetenr la cordenada de donde se encuntra la punta de nuestro dedo INDICE..
        ## okey en la linea 234, tenemos un metodo que se llama "get_finger_tip", si le pasamos como parametro que queremos el indice 2-- es decir el dedo indice
        indice_tip = self.get_finger_tip(hand_landmarks, 2)##  tip_id = finger_index * 4 -> esto esta en la linea 234,  2*4 = 8 el landamarks 8
        # punta del dedo indice, ademas es lindo saber que es un diccionario creado con 4 dedos, si multiplicaras por 3*4 tendrias el landamarks del dedo medio
        x, y = indice_tip## estriamos obteniendo las cordenadas de la punta del dedo inidce

        # También obtener la segunda falange (punto 7) para ver si está "señalando"
        h, w = self.height, self.width ## pasamos el ancho y largo de la pantalla
        indice_mid = hand_landmarks.landmark[7] ## y nos traemos a la falange del dedo indice
        x2 = int(indice_mid.x * w) ## al multiplicarla por las medidas de w (horizontal) tendrias las codenadas en pantalla de ese landamarks(falange del dedo indice ) que equivale a 7 en x
        y2 = int(indice_mid.y * h)## al multiplicarla por las medidas de h (vertical) tendrias las codenadas en pantalla de ese landamarks(falange del dedo indice ) que equivale a 7 en y

        # Verificar que el dedo está extendido y señalando hacia arriba
        # La punta (y) debe estar más arriba que la falange (y2)
        # y2 - y > 20 significa que hay al menos 20 píxeles de diferencia
        dedo_extendido = (y2 - y) > 20

        ## decimos que si cumple "dedo_extendido", y que ademas:
        #palette_y <= y : El dedo está ABAJO del borde superior de la paleta
        ##"palette_y + palette_height": El dedo está ARRIBA del borde inferior de la paleta
        if dedo_extendido and palette_y <= y <= palette_y + palette_height: ## esta linea quiere decir que mientras y (la punta del dedo indice) este en una cordenada:
            ## donde el no sobrepase el borde superior de la paleta, y ademas no sobrosalga del limite superior de la paleta... entonces:

            # Recorremos todos los colores para ver si el dedo tocó alguno:
            #para ello vamos a recorrer el diccionario "self.colors" -> linea 70
            for i in range(len(self.colors)):
                color_x = start_x + i * color_width ## calculamos cada posisción de cada color en x
                if color_x <= x <= color_x + color_width - 5: ## aqui verificamos si el dedo toco el color, en el x
                    self.current_color_index = i ### aqui vamos a igualar el color que tomo el dedo a " self.current_color_index " que esta establecida con color verde por default en la linea 92
                    self.color = list(self.colors.values())[i]
                    return True  # Seleccionó color

            # Botón de limpiar... al estra dnetro del if linea 368.. calculamos en que posisción del eje x se encunetra el boton de borrar
            clear_x = start_x + len(self.colors) * color_width
            if clear_x <= x <= clear_x + color_width - 5: ## verificamos si el dedo toco el boton de borrar, en dado caso de que sea verdadero
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)##se inicializa la pantalla
                return True  # Limpió el canvas

        return False  # No seleccionó nada

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) ## configuración de la camara, el 0 nos indica que es la camra que viene por defecto
        ## aqui tenemos la configuración para la resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width) # de ancho queremos 1280 pixeles
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height) ### y de largo queremos 720
        ## estas lineas anteriores es como si encendieramos nuestra camarita

        print("Air Painter - Dibuja con tus dedos")
        print("Instrucciones:")
        print("- Dedo índice levantado: Dibujar")
        print("- Dedo medio levantado: Cambiar tamaño del pincel")
        print("- Dedo índice + medio: Pausar/Reanudar dibujo")
        print("- Señalar a la paleta de colores: Cambiar color")
        print("- Presiona 'c' para limpiar")
        print("- Presiona 's' para guardar dibujo")
        print("- Presiona 'ESC' para salir") ## todas estas sin instruciones de uso, que se van a impirmir en consola

        while True: ## este bucle se ejecuta al rededor de 30 veces por segundo (30 FPS)
            ret, frame = cap.read()##ret es un valor booleando que nos indica si se pudo leer el frame o no
            ## frame, vendría siendo la imagen capturada por la camra... claro en formato "BGR" BLUE GREEN RED
            if not ret: ## SI NO SE PUDO LEER EL FRAM
                break ## SALIMOS DEL BUCLE
            ###"cv2.flip(frame, 1)" Voltea la imagen horizontalmente, hmm haciendo como efecto espejo
            frame = cv2.flip(frame, 1) ## el paramtero 1 nos indica que el volteo va a ser en horizontal
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) ## "cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)" convierte la imagen de BGR (formato de OpenCV) a RGB (formato que entiende MediaPipe)
            results = self.hands.process(frame_rgb) ##MediaPipe analiza la imagen y detecta manos

            # Dibujar paleta de colores en la pantalla
            frame = self.draw_color_palette(frame)

            if results.multi_hand_landmarks: ## en dado caso de que si se haya detectado una mano
                # IMPORTANTE: Usamos enumerate para obtener el índice y poder acceder a multi_handedness
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks): ## verficamos de que mano estamos hablando de si la derecha o de la segunda por medio de "idx"
                    ## que para la 1era mano tenemos el indice 0 y para la segunda mano el indice 1

                    # Obtener la orientación de la mano (Left/Right) - CORREGIDO, para saber como interpretar el dedo pulgar de la mano
                    handedness = results.multi_handedness[idx].classification[0].label

                    # Detectar gestos (ahora con handedness)
                    dedos = self.detectar_gestos(hand_landmarks, handedness)

                    # Obtener coordenadas del dedo índice, mandamos a llamar a la función "self.get_finger_tip(hand_landmarks, 2)" le pasamos las landamarks como parametro y el indice 2
                    indice_tip = self.get_finger_tip(hand_landmarks, 2)  # Índice
                    medio_tip = self.get_finger_tip(hand_landmarks, 3)  # Medio
                    # Obtener coordenadas del dedo medio, mandamos a llamar a la función "self.get_finger_tip(hand_landmarks, 3)" le pasamos las landamarks como parametro y el indice 3

                    # Verificar selección de color (cuando el dedo índice señalaa un cuadrito de color) ooo si solo esta dibujando
                    if dedos[1] and not dedos[2]:  # Solo índice levantado
                        # Ahora pasamos hand_landmarks completo
                        if self.check_color_selection(hand_landmarks):
                            # decimos que si seleccionó color o limpió la pantalla, reseteamos prev_point
                            # Esto evita que se dibuje una línea al salir de la paleta
                            self.prev_point = None ## se restablce el putno anterior, para que no se haga una linea seguida del anterior color
                        else: ## decimos que si no se seleciono ningun colo ni tampoco se borro/limpio la pantalla entonces
                            # Dibujar
                            if self.prev_point is not None: ## aqui indicamos que el punto anterior esta limpio
                                ## "cv2.line(self.canvas" -> en el lienzo
                                cv2.line(self.canvas, self.prev_point, indice_tip,
                                         self.color, self.brush_size)
                                ##"cv2.line(frame," -> en el frame actual
                                cv2.line(frame, self.prev_point, indice_tip,
                                         self.color, self.brush_size)
                            self.prev_point = indice_tip ## guardamos el punto anterior actual

                    # Controlar tamaño del pincel con dedo medio
                    if len(dedos) > 2 and dedos[2] and not dedos[1]:## decimos que sie el solo tenemos el dedo medio levantado y no otro dedo
                        self.brush_size += 1 ## entonces que nos aumente el taño del pincel en uno

                        if self.brush_size > 50: ## pero qeu una vez llegando a 50 pixeles
                            self.brush_size = 5 ## vuelva a restablecer el grosor en 5

                        cv2.putText(frame, f"Pincel: {self.brush_size}", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2) ## aqui al alzar el dedo en la punta del dedo indice
                        ## podriamso ver un pequeño letrerito indicandonos como va aumnetando el groso del pincel dinamicamente

                    # Cambiar modo (pausar dibujo) con índice + pulgar
                    if len(dedos) > 2 and dedos[1] and dedos[2]: ## cuando tenemos los 2 dedos levantados
                        self.prev_point = None ## decimos que el último punto se restablece en none, lo que indica que ya estamos en modo pausa y se deja de dibujar
                        cv2.putText(frame, "MODO PAUSA", (10, 150), ## aqui colocamos en pantalla un letrerito que diga "modo pausa"
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) ## aqui le estamos dando formato al letrerito
                    elif not dedos[1]: ## en dado caso de que el dedo el dedo pulgar no este levantado
                        self.prev_point = None

                    # Marcar la punta del dedo índice
                    cv2.circle(frame, indice_tip, self.brush_size, self.color, -1)

                    # Mostrar gesto actual
                    gesto_texto = "Dibujando" if dedos[1] and not dedos[2] else "Pausa"
                    cv2.putText(frame, f"Modo: {gesto_texto}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Dibujar landmarks de la mano
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks,
                                                   self.mp_hands.HAND_CONNECTIONS)

            # Combinar canvas con el frame
            result = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)

            # Mostrar instrucciones
            cv2.putText(result, "Presiona 'c' limpiar | 's' guardar | 'ESC' salir",
                        (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Air Painter - Dibuja con tus dedos', result)

            key = cv2.waitKey(1) & 0xFF ### aqui estamos esperando a que el usuario presione una tecla por un milesegundo a que el usuario presione una tecla...
           # entonces dependiendo de que valor tome la tecla que esperamos vamos a hacer las siguientes cosas
            if key == 27:  # es la tecla ESC
                break ## se rompe el bucle y salimos del programa
            elif key == ord('c'): ## si toma el valor de la tecla c, restablecemos el canvas, osea limpiamos la paantall
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8) ## esta sería otra forma de limpiar la pantalla sin señalar con el dedo
                print("Canvas limpiado") ## imprimimos en consola
            elif key == ord('s'): ## aqui si presionamos la teclas s decimos que queremos guardar el dibujo pedorro que hicimos
                filename = "mi_dibujo.png" ## toma un nombre
                cv2.imwrite(filename, self.canvas)
                print(f"Dibujo guardado como '{filename}'") ## se impirme en consola que el dibujo a sido guardado

        cap.release()### aqui liberamos la camara y apagamos
        cv2.destroyAllWindows() ## aqui ya cerramos todas las ventanitas que llego a abrir "OpenCV"


# Versión simplificada para comenzar rápido... sd
class SimpleAirPainter:
    """Versión más simple para empezar rápidamente"""

    def __init__(self):
        self.mp_hands = hands ## guardamos herramientas de construción de manos
        self.mp_drawing = drawing_utils ## guardamos las herraminetas para dibujar los puntitos en la mano

        self.hands = self.mp_hands.Hands( ## aqui creamos el detector de amnos, pero solamente tendra la configuración de
            min_detection_confidence=0.7 ## confiar 0.7 en que lo que ve es una mano
        )

        self.canvas = None ## el lienzo lo vamos a crear cuando tengamos las dimensiones, por eso esta inicializado en "None"
        self.prev_point = None ## aun no hay punto anterior
        self.drawing = False ## y pues aun no dibujamos

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) ## aqui tenemos la configuración para la camara

        while True:
            ret, frame = cap.read() ##ret es un valor booleando que nos indica si se pudo leer el frame o no
            ## frame, vendría siendo la imagen capturada por la camra... claro en formato "BGR" BLUE GREEN RED
            if not ret: ## si no se pudo leer el frame
                break ## se rompe el bucle

            frame = cv2.flip(frame, 1) ## hacemos el efecto espejo
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)## creamos lienzo en negro del tamaño del frame

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)## "cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)" convierte la imagen de BGR (formato de OpenCV) a RGB (formato que entiende MediaPipe)
            results = self.hands.process(frame_rgb)##MediaPipe analiza la imagen y detecta manos

            if results.multi_hand_landmarks: ## aqui verificamos si existe una mano en la pantalla
                hand = results.multi_hand_landmarks[0] ## en la version simple vamos a tomar solo una mano, por eso solo tenemos el indice 0 y ya

                # Obtener coordenadas del dedo índice
                h, w, _ = frame.shape

                index_landmark = hand.landmark[8] ## obtenemos el landamarks de la punta del dedo indice, osea el 8

                x = int(index_landmark.x * w)## y aqui multiplicamos con el ancho
                y = int(index_landmark.y * h) ## y aqui por el largo

                # Dedo medio para saber si dibujar

                middle_tip = hand.landmark[12] ## aqui obtenemos el landamarks del dedo medio, osea el 12

                drawing = middle_tip.y > index_landmark.y  # Decimos que si "middle_tip" en y es mayor a index_landmark.y (el dedo indice )
                ## siguiendo la logica de que mientras mas abajo este el dedo es mas grande y que mientras mas arriba este el dedo es ma pequeño

                if drawing: ## si se cumple "drawing"
                    if self.prev_point:
                        cv2.line(self.canvas, self.prev_point, (x, y), (0, 255, 0), 5)  # Dibujar línea en el canvas (permanente)
                        cv2.line(frame, self.prev_point, (x, y), (0, 255, 0), 5)
                    self.prev_point = (x, y)
                else:
                    self.prev_point = None ## y decimos que si el dedo medio esta arriba no dibujamos nada y por eso se restablece el punto anterior

                # Mostrar punto de dibujo
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Combinar
            result = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
            cv2.imshow('Air Painter Simple', result) ## aqiu tenemos que vamos a combinar 70% camara y 30% dibujo

            if cv2.waitKey(1) & 0xFF == 27:## aquí como estamos en modo simple, decimos que si presionamos la tecla  ESC
                break ## se rompe el bucle y se cierra el programita

        cap.release() ### aqui liberamos la camara y apagamos
        cv2.destroyAllWindows()  ## aqui ya cerramos todas las ventanitas que llego a abrir "OpenCV"

#######################################################################################################
## esto es lo 1ero que vamos a ver en consola para elegir la versión simple o la version dificil
if __name__ == "__main__":
    # Este código solo se ejecuta si corro el archivo directamente
    print("Elige una versión:")
    print("1. Versión Completa (colores, tamaños, paleta)")
    print("2. Versión Simple (para empezar rápido)")

    opcion = input("Opción (1 o 2): ").strip() ## aqui ya capturamos la opción que elegio el usuario

    if opcion == "1": ## si selecionamos la versin 1, tendremos la versión completa
        painter = AirPainter() ## Creamos una instancia de la versión completa
        painter.run() ## Ejecutamos el método run()
    else: ## si se seleciona la version 2
        painter = SimpleAirPainter() #Creamos una instancia de la versión completa
        painter.run() ## ejecutasmo el metodo run