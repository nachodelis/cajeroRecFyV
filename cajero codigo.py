import speech_recognition as sr
import mysql.connector
import random
from datetime import date
import cv2
import os
import imutils
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import pyttsx3

engine = pyttsx3.init()
def print_and_speak(message):
    print(message)
    engine.say(message)
    engine.runAndWait()
def speak(message):
    engine.say(message)
    engine.runAndWait()
def speakargs(*args):
    engine = pyttsx3.init()
    message = ' '.join(map(str, args))
    print(message)
    engine.say(message)
    engine.runAndWait()
cnn=mysql.connector.connect(host="localhost",user="root",database="cajero")
r = sr.Recognizer()
cur=cnn.cursor()
def clave(dni):  
    speak('Introduzca su pin')
    pin = input('Introduzca su pin: ')
    SQL2="SELECT cuentas.clave FROM cuentas INNER JOIN usuarios ON cuentas.id_usuario = usuarios.id WHERE usuarios.id = (%s)"
    cur.execute(SQL2,(dni,))
    datos2 = cur.fetchall()
    datos2 = str(datos2).replace(",", "")
    datos2 = str(datos2).replace("[(", "")
    datos2 = str(datos2).replace(")]", "")
    datos2 = str(datos2).replace("'", "")
    if datos2 == pin:
        print_and_speak('Clave correcta, comencemos')
        return True
    else:
        print_and_speak('Clave incorrecta')
        return False
    
def empezar():
    speak('Introduzca su dni: ') 
    dni = input('Introduzca su dni: ')         
    cur=cnn.cursor()
    SQL = "SELECT * FROM usuarios WHERE id = %s"
    cur.execute(SQL,(dni,))
    datos = cur.fetchall()
    IDD = datos[0][0]
    if datos is not None:
        print_and_speak('Todo correcto')
    else:
        print_and_speak('No hay ningún usuario perteneciente a este banco')
        return
    clavePin = clave(dni)
    reconocimiento = usarmodelo(dni)
    if clavePin == False or reconocimiento == 'no':
        return
    


    print_and_speak('Tiene las siguientes opciones:  ')
    print_and_speak('1. Consultar cuenta')
    print_and_speak('2. Ingresar dinero')
    print_and_speak('3. Retirar dinero')
    print_and_speak('4. Transferencia')
    print_and_speak('5. Llamar a operador')
    print_and_speak('6. Cerrar sesión')
    cerrar = []
    problema = []
    while True:
        audio=r.listen(source)
        try:
                text = r.recognize_google(audio, language="es-ES")
                print_and_speak('Usted ha dicho: {}'.format(text))
                if "consultar" in text:
                    print_and_speak('Vamos a consultar el saldo de su cuenta: ')
                    consultar(dni)
                    print_and_speak('Cuenta consultada')
                elif "ingresar" in text:
                    ingresar(dni)
                    reconocimiento = usarmodelo(dni)
                    if reconocimiento == 'no':
                       problema = 'True'
                elif "retirar" in text:
                    retirar(dni)
                    reconocimiento = usarmodelo(dni)
                    if reconocimiento == 'no':
                       problema = 'True'
                elif "transferencia" in text:
                    transferencia(dni)
                    reconocimiento = usarmodelo(dni)
                    if reconocimiento == 'no':
                       problema = 'True'
                elif "cerrar" in text:
                    cerrar = 'True'
                elif "operador" in text:
                    print_and_speak('Llamando a operador')
                    cerrar = 'True'
                else:
                    print_and_speak('Esto no es una operación, por favor, ¿podría volver a repetirlo?')
        except:
            print_and_speak('No te he entendido, por favor ¿podría volver a repetirlo?')
        if cerrar == 'True':
            print_and_speak('Hola soy tu asistente por voz, para empezar, si desea una explicación, diga "Ayuda", si desea introducir su tarjeta, diga "Empezar", si desea salir del programa, diga "Salir"')
            return
        elif problema == 'True':
            Var = 0
            return Var

         

def ingresar(IDD):
    speak('¿Cuanto dinero desea ingresar?')
    dinero = input('¿Cuanto dinero desea ingresar?: ')
    
    dinero = int(dinero)
    SQL4="UPDATE cuentas SET saldo = saldo+(%s) WHERE id_usuario = (%s)"
    cur.execute(SQL4,(int(dinero),IDD))
    SQL3="SELECT cuentas.saldo FROM cuentas INNER JOIN usuarios ON cuentas.id_usuario = usuarios.id WHERE usuarios.id = (%s)"
    cur.execute(SQL3,(IDD,))
    datos3 = cur.fetchall()
    datos3 = str(datos3).replace("[(Decimal('", "")
    datos3 = str(datos3).replace("'),)]", "") 
    speakargs('Su dinero disponible en la cuenta es de: ',datos3)
    print_and_speak('¿Qué otra consulta desea realizar?')
    return

def retirar(IDD):
    speak('¿Cuanto dinero desea retirar?')
    dinero = input('¿Cuanto dinero desea retirar?: ')
    
    dinero = int(dinero)
    SQL5="UPDATE cuentas SET saldo = saldo-(%s) WHERE id_usuario = (%s)"
    cur.execute(SQL5,(int(dinero),IDD))
    SQL3="SELECT cuentas.saldo FROM cuentas INNER JOIN usuarios ON cuentas.id_usuario = usuarios.id WHERE usuarios.id = (%s)"
    cur.execute(SQL3,(IDD,))
    datos3 = cur.fetchall()
    datos3 = str(datos3).replace("[(Decimal('", "")
    datos3 = str(datos3).replace("'),)]", "") 
    speakargs('Su dinero disponible en la cuenta es de: ',datos3)
    print_and_speak('¿Qué otra consulta desea realizar?')
    return
def consultar(IDD):
    SQL3="SELECT cuentas.saldo FROM cuentas INNER JOIN usuarios ON cuentas.id_usuario = usuarios.id WHERE usuarios.id = (%s)"
    cur.execute(SQL3,(IDD,))
    datos3 = cur.fetchall()
    datos3 = str(datos3).replace("[(Decimal('", "")
    datos3 = str(datos3).replace("'),)]", "")  
    speakargs('Su dinero disponible en la cuenta es de: ',datos3)
    print_and_speak('¿Qué otra consulta desea realizar?')
def transferencia(IDD):
    speak('¿Cuánto dinero desea transferir?')
    dinero = input('¿Cuánto dinero desea transferir?: ')
    dinero = int(dinero)
    print_and_speak('¿A quien desea transferir el dinero?')
    nombre2 = input('Nombre: ')
    apellido2 = input('Apellido: ')
    cur=cnn.cursor()
    SQL = "SELECT * FROM usuarios WHERE Nombre = (%s) AND Apellido = (%s)"
    cur.execute(SQL,(nombre2,apellido2))
    datos10 = cur.fetchall()
    IDD2 = datos10[0][0]
    if datos10 is not None:
        print_and_speak('Usuario registrado')
    else:
        print_and_speak('Usuario no registrado')
        return
    speak('Introduzca de nuevo el pin para confirmar la operación: ')
    pinn = input('Introduzca de nuevo el pin para confirmar la operación: ')
    SQL2="SELECT cuentas.clave FROM cuentas INNER JOIN usuarios ON cuentas.id_usuario = usuarios.id WHERE usuarios.id = (%s)"
    cur.execute(SQL2,(IDD,))
    datos2 = cur.fetchall()
    datos2 = str(datos2).replace(",", "")
    datos2 = str(datos2).replace("[(", "")
    datos2 = str(datos2).replace(")]", "")
    datos2 = str(datos2).replace("'", "")
    if datos2 == pinn:
        print_and_speak('Clave correcta, operación realizada')
        SQL5="UPDATE cuentas SET saldo = saldo-(%s) WHERE id_usuario = (%s)"
        cur.execute(SQL5,(int(dinero),IDD))
        SQL3="SELECT cuentas.saldo FROM cuentas INNER JOIN usuarios ON cuentas.id_usuario = usuarios.id WHERE usuarios.id = (%s)"
        cur.execute(SQL3,(IDD,))
        datos3 = cur.fetchall()
        datos3 = str(datos3).replace("[(Decimal('", "")
        datos3 = str(datos3).replace("'),)]", "")
        speakargs('Su dinero disponible en la cuenta es de: ',datos3)
        print_and_speak('¿Qué otra consulta desea realizar?')
        SQL4="UPDATE cuentas SET saldo = saldo+(%s) WHERE id_usuario = (%s)"
        cur.execute(SQL4,(int(dinero),IDD2))
        
    else:
        print_and_speak('Clave incorrecta')
        return    

def registrar():
    print_and_speak('Para registrarse, le tomaremos sus datos y el sistema tomará un video suyo para poder utilizar el reconocimiento facial')
    cur=cnn.cursor()
    speak('Introduzca su dni')
    idd = input('Introduzca su dni: ')
    speak('Introduzca su dirección')
    direccion = input('Introduzca su direccion: ')
    speak('Introduzca su nombre')
    nombre = input('Introduzca su nombre: ')
    speak('Introduzca su apellido')
    apellido = input('Introduzca su apellido: ')
    speak('Introduzca su telefono')
    telefono = input('Introduzca su telefono: ')
    speak('Introduzca su correo')
    correo = input('Introduzca su correo:  ')
    SQLR="INSERT INTO usuarios (id, nombre, apellido, direccion, telefono, correo) VALUES ((%s), (%s), (%s), (%s), (%s), (%s))"
    cur.execute(SQLR,(idd,nombre,apellido,direccion,telefono,correo))
    print_and_speak('Usuario registrado con exito')
    print_and_speak('Procedemos a crear tu cuenta y a enviarte los datos a tu correo electrónico: ')
    numero_cuenta = str(random.randint(100000000, 999999999))
    saldo = 0
    fecha_creacion = date.today().strftime('%Y-%m-%d')
    speak('Introduzca la clave que desea utilizar, debe ser de 4 dígitos')
    clave = input('Introduzca la clave que desea utilizar, debe ser de 4 dígitos:  ')
    SQLR = "INSERT INTO cuentas (id_usuario, numero_cuenta, saldo, fecha_creacion, clave) VALUES (%s, %s, %s, %s, %s)"
    values = (idd, numero_cuenta, saldo, fecha_creacion, clave)
    cur.execute(SQLR, values)
    print_and_speak('Procedemos al reconocimiento facial, primero le tomaremos las fotos: ')
    tomarfotos(idd)
    entrenarmodelo(idd)
    ruta_imagenes = r'C:\Users\PYTHON.LAPTOP-CI80DL17\Desktop\UFV\TFG\codigo\reconF\personas'
    ruta_imagenes = os.path.join(ruta_imagenes, idd)
    ruta_imagenes2 = os.path.join(ruta_imagenes, 'imagenes')
    ruta_modelo = os.path.join(ruta_imagenes, 'modelo')
    sql = "INSERT INTO reconocimiento_facial (id_usuario, modelo, ruta_imagen) VALUES (%s, %s, %s)"
    values = (idd,ruta_modelo, ruta_imagenes2,)
    cur.execute(sql, values)
    return
    

def usarmodelo(idd):
    dataPath = r'C:\Users\PYTHON.LAPTOP-CI80DL17\Desktop\UFV\TFG\codigo\reconF\personas'
    imagePaths = os.listdir(dataPath)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer_path = os.path.join(dataPath, idd)
    face_recognizer_path = os.path.join(face_recognizer_path, 'modelo')
    face_recognizer_path = os.path.join(face_recognizer_path, 'modeloLBPHFace.xml')
    face_recognizer.read(face_recognizer_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()
    no_face_counter = 0
    detected_my_face = False  # Variable para realizar seguimiento de la detección de tu rostro
    my_face_counter = 0
    my_face_pos = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        no_face_counter += 1


        if len(faces) > 0:
            no_face_counter = 0
            
            # Reiniciar la variable detected_my_face en cada iteración
            
            for (x, y, w, h) in faces:
                rostro = auxFrame[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro)
                

                if result[1] < 70:
                    # Si se detecta tu rostro, actualiza la variable detected_my_face a True
                    my_face_counter = 0
                    detected_my_face = True
                    break
                
        
        elapsed_time = time.time() - start_time
        
        if detected_my_face:
            start_time = time.time()
            detected_my_face = False  # Reiniciar el tiempo cuando se detecta tu rostro
            my_face_pos += 1
        else:
            my_face_counter += 1
        
        if no_face_counter > 100 or my_face_counter > 100:
            print_and_speak("No se detectó tu rostro durante su cara. Cerrando la cámara.")
            face = 'no'
            break 
        elif my_face_pos > 50:
            print_and_speak("Se detecto su cara")
            face = 'si'
            break 

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 70:
                cv2.putText(frame, idd, (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('Camara', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return face





def tomarfotos(idd):
    
    dataPath = r'C:\Users\PYTHON.LAPTOP-CI80DL17\Desktop\UFV\TFG\codigo\reconF\personas'
    personPath = os.path.join(dataPath, idd)
    imagePath =os.path.join(personPath, 'imagenes')

    if not os.path.exists(personPath):
        print('Carpeta creada:', personPath)
        os.makedirs(personPath)
        os.makedirs(imagePath)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(imagePath, f'rostro_{count}.jpg'), rostro)
            count += 1

        # Mostrar el fotograma en una ventana emergente
        cv2.imshow('Camara', frame)
        if cv2.waitKey(1) == 27 or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()

    return

def entrenarmodelo(idd):
    dataPath = r'C:\Users\PYTHON.LAPTOP-CI80DL17\Desktop\UFV\TFG\codigo\reconF\personas'  # Cambia a la ruta donde hayas almacenado los datos
    personPath = os.path.join(dataPath, idd)  # Cambia 'nombre_carpeta' por el nombre de la carpeta de entrenamiento
    personPath2=os.path.join(personPath, 'imagenes')
    label = 1
    facesData = []
    labels = []
    print(personPath)
    for imagen in os.listdir(personPath2):
            print('Rostros:', imagen)
            labels.append(label)
            facesData.append(cv2.imread(os.path.join(personPath2, imagen)))
    print(labels)
    print(facesData)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Entrenando...")
    for i, faceData in enumerate(facesData):
        gray = cv2.cvtColor(faceData, cv2.COLOR_BGR2GRAY)
        facesData[i] = cv2.resize(gray, (150, 150), interpolation=cv2.INTER_CUBIC)


    face_recognizer.train(facesData, np.array(labels))

    modelPath = os.path.join(personPath, 'Modelo')
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    # Almacenando el modelo obtenido
    modelFile = os.path.join(modelPath, 'modeloLBPHFace.xml')
    face_recognizer.write(modelFile)
    print("Modelo almacenado en:", modelFile)
    print(dir(cv2.face))
    return


    

Var = []
with sr.Microphone() as source:
    print_and_speak('Hola soy tu asistente por voz, para empezar,diga registrarse,si ya está registrado, diga empezar, si desea una explicación, diga "Ayuda", si desea salir del programa, diga "Salir"')
    salir=[]
    problema = []
    while True:
        audio=r.listen(source)
        try:
            text = r.recognize_google(audio, language="es-ES")
            print_and_speak('Has dicho: {}'.format(text))
            if "empezar" in text:
                Var = empezar()
            elif "salir" in text:
                salir = 'True'
            elif "ayuda" in text:
                print_and_speak('Este cajero te permite realizar todas las operaciones que necesites, sin necesidad de utilizar la pantalla, solo hablando al microfono, si en algún momento necesita ayuda telefónica, solo diga "Hablar con operador"')
            elif "operador" in text:
                print_and_speak('Llamando a operador')
                salir = 'True'
            elif "registrarse" in text:
                registrar()
            else:
                print_and_speak('Eso no es una opción, ¿podría volver a repetirlo?')
        except:
            print_and_speak('No te he entendido, ¿podría volver a repetirlo?')  
        if salir == 'True' or problema == 'True':
            print_and_speak('¡Que tenga un buen día!')
            cnn.commit()
            cur.close()
            cnn.close()
            break
        elif Var == 0:
            break


            





