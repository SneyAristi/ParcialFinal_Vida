import numpy as np
import random
import matplotlib.pyplot as plt
from numba import jit
import turtle
from PIL import Image
import os

class AlgoritmoGenetico:
    def __init__(self, tamano_poblacion, tamano_cromosoma, tasa_cruce, tasa_mutacion, num_elitismo, axioma, iteraciones):
        self.tamano_poblacion = tamano_poblacion
        self.tamano_cromosoma = tamano_cromosoma
        self.tasa_cruce = tasa_cruce
        self.tasa_mutacion = tasa_mutacion
        self.num_elitismo = num_elitismo
        self.poblacion = self.generar_poblacion_inicial(tamano_poblacion, tamano_cromosoma)
        self.historial_fitness = []
        self.mejores_individuos = []  # Lista para almacenar los mejores individuos de cada generación
        self.iteraciones= iteraciones
        self.axioma= axioma

    @staticmethod
    @jit(nopython=True)
    def generar_poblacion_inicial(tamano_poblacion, tamano_cromosoma):
        poblacion = np.empty((tamano_poblacion, tamano_cromosoma), dtype=np.int64)
        for i in range(tamano_poblacion):
            for j in range(tamano_cromosoma):
                poblacion[i, j] = np.random.randint(0, 6)  # Generar números entre 0 y 5
        return poblacion

    def evaluar_fitness(self, individuo):
        ar = Arbol()
        regla = ''.join(ar.num_a_lenguaje_individuo(individuo))
        reglas = {'F': 'FF', 'G': regla}

        # Axioma inicial
        cadena = str(self.axioma)

        # Generar la cadena del L-System
        for _ in range(self.iteraciones):
            cadena = ''.join(reglas.get(simbolo, simbolo) for simbolo in cadena)

        # Penalizar si la cadena no contiene todos los símbolos esenciales
        simbolos_requeridos = {'F', 'G', '[', ']', '+', '-'}
        if not simbolos_requeridos.issubset(set(cadena)):
            return 1  # Fitness mínimo
        
        if not cadena.startswith('F'):
            return 1  # Penalización si no empieza con 'F'

        # Penalizar desequilibrio de corchetes
        balance = 0
        for simbolo in cadena:
            if simbolo == '[':
                balance += 1
            elif simbolo == ']':
                balance -= 1
            if balance < 0:
                return 1  # Penalización por corchetes desbalanceados
        if balance != 0:
            return 1
        if cadena.count('[') < 4:
            return 1  # Penalización si no hay al menos 3 corchetes


        # Penalizar redundancia (repeticiones consecutivas)
        repeticiones_consecutivas = sum(
            1 for i in range(1, len(cadena)) if cadena[i] == cadena[i - 1]
        )
        if repeticiones_consecutivas > len(cadena) * 0.1:
            return 1

        # Premiar diversidad y estructura
        total_simbolos = len(cadena)
        uso_F = cadena.count('F') / total_simbolos
        uso_G = cadena.count('G') / total_simbolos
        uso_corchetes = (cadena.count('[') + cadena.count(']')) / total_simbolos
        diversidad = len(set(cadena))  # Número de símbolos únicos

        # Penalizar cadenas con un símbolo dominante
        if max(uso_F, uso_G, uso_corchetes) > 0.8:
            return 1

        # Premiar tamaño razonable y diversidad
        tamaño_final = len(cadena)
        if tamaño_final > 25000:
            return 1  # Penalización por exceso de tamaño

        # Fitness basado en crecimiento, diversidad y balance de símbolos
        fitness = (tamaño_final * diversidad) / (repeticiones_consecutivas + 1)
        return max(1, fitness)





    def seleccion_torneo(self, tamanio_torneo=3):
        seleccionados = np.random.randint(0, self.tamano_poblacion, size=tamanio_torneo)
        torneo = [self.poblacion[i] for i in seleccionados]
        torneo_fitness = [self.evaluar_fitness(ind) for ind in torneo]
        mejor_individuo = torneo[np.argmax(torneo_fitness)]
        return mejor_individuo


   
    def cruce_dos_puntos(self, padre1, padre2):
        punto1 = random.randint(1, len(padre1) - 2)  # Asegurarse de que los puntos no estén al final
        punto2 = random.randint(punto1 + 1, len(padre2) - 1)
        
        hijo1 = np.concatenate((padre1[:punto1], padre2[punto1:punto2], padre1[punto2:]))
        hijo2 = np.concatenate((padre2[:punto1], padre1[punto1:punto2], padre2[punto2:]))
        
        return hijo1, hijo2


    @staticmethod
    @jit(nopython=True)
    def mutacion_inversion(individuo, tasa_mutacion):
        if random.random() < tasa_mutacion:
            inicio = random.randint(0, len(individuo) - 1)
            fin = random.randint(inicio, len(individuo) - 1)
            individuo[inicio:fin] = individuo[inicio:fin][::-1]
        return individuo

    def nueva_generacion(self, generacion):
        # Ordenar la población por fitness y seleccionar los élite
        fitness_poblacion = [(ind, self.evaluar_fitness(ind)) for ind in self.poblacion]
        fitness_poblacion.sort(key=lambda x: x[1], reverse=True)
        elite = [ind for ind, _ in fitness_poblacion[:self.num_elitismo]]

        # Guardar el mejor individuo de la generación actual
        if generacion % 10 == 0:
            self.mejores_individuos.append(fitness_poblacion[0][0])

        # Generar nueva población con los procesos de cruce y mutación
        nueva_poblacion = elite  # Los élite se añaden directamente a la nueva población
        while len(nueva_poblacion) < self.tamano_poblacion:
            # Selección de padres por torneo
            padre1 = self.seleccion_torneo()
            padre2 = self.seleccion_torneo()

            # Cruce de dos puntos con la probabilidad de cruce
            if random.random() < self.tasa_cruce:
                hijo1, hijo2 = self.cruce_dos_puntos(padre1, padre2)
            else:
                hijo1, hijo2 = padre1, padre2

            # Mutación
            nueva_poblacion.append(self.mutacion_inversion(hijo1, self.tasa_mutacion))
            if len(nueva_poblacion) < self.tamano_poblacion:
                nueva_poblacion.append(self.mutacion_inversion(hijo2, self.tasa_mutacion))

        self.poblacion = np.array(nueva_poblacion[:self.tamano_poblacion])

    def ejecutar(self, generaciones):
        generacion =1
        terminar=0
        acabar = False
        while acabar == False:
            fitness_values = np.array([self.evaluar_fitness(ind) for ind in self.poblacion])
            promedio = np.mean(fitness_values)
            varianza = np.var(fitness_values)

            print(f"\nGeneración {generacion + 1}")
            print(f"Fitness promedio: {promedio}")
            print(f"Varianza del fitness: {varianza}")
            self.historial_fitness.append(fitness_values)  # Guardar los valores de fitness de la generación actual

            self.nueva_generacion(generacion)
            mejor = max(self.poblacion, key=self.evaluar_fitness)
            print(f"Mejor individuo: {mejor}, Fitness: {self.evaluar_fitness(mejor)}")
            if  self.evaluar_fitness(mejor) == 10000:
                terminar += 1
                self.mejores_individuos.append(mejor)
                
                if terminar == 2:  # Si el objetivo se mantiene dos generaciones consecutivas
                    print(f"Objetivo alcanzado en la generación {generacion + 1}")
                    acabar = True
            if generacion >= generaciones:
                print("Límite de generaciones alcanzado.")
                break
            generacion+=1

        self.graficar_boxplot()
        self.imprimir_mejores_individuos()
        lenguajes= self.imprimir_individuos_lenguaje()
        return lenguajes

    def graficar_boxplot(self):
        plt.figure(figsize=(10, 6))
        plt.boxplot(self.historial_fitness, positions=np.arange(1, len(self.historial_fitness) + 1))
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Distribución del Fitness de la Población a través de Generaciones')
        plt.xticks(np.arange(1, len(self.historial_fitness) + 1))
        plt.show()
    
    def imprimir_individuos_lenguaje(self):
        arbol= Arbol()
        lenguajes=arbol.num_a_lenguaje(self.mejores_individuos)
        for i,lenguaje in enumerate(lenguajes):
            print(f"Generación {i + 1}: {lenguaje}")
        return lenguajes

    def imprimir_mejores_individuos(self):
        print("\nMejores individuos por generación:")
        for i, ind in enumerate(self.mejores_individuos):
            print(f"Generación {i + 1}: {ind}")
            
class Arbol:
    @staticmethod    
    def lenguaje_a_num(cadena):
        numeros=[]
        for cad in cadena:
            if cad=="G":
                numeros.append(0)
            if cad=="F":
                numeros.append(1)
            if cad=="+":
                numeros.append(2)
            if cad=="-":
                numeros.append(3)
            if cad=="[":
                numeros.append(4)
            if cad=="]":
                numeros.append(5)
        return numeros
    
    @staticmethod
    def num_a_lenguaje(cadena):
        lenguajes=[]
        numeros=[]
        for pal in cadena:
            for cad in pal:
                if cad==0:
                    numeros.append("G")
                if cad==1:
                    numeros.append("F")
                if cad==2:
                    numeros.append("+")
                if cad==3:
                    numeros.append("-")
                if cad==4:
                    numeros.append("[")
                if cad==5:
                    numeros.append("]")
            lengu = ''.join(map(str, numeros))
            numeros.clear()
            lenguajes.append(lengu)
        return lenguajes
    
    @staticmethod
    def num_a_lenguaje_individuo(cadena):
        numeros=[]
        for cad in cadena:
            if cad==0:
                numeros.append("G")
            if cad==1:
                numeros.append("F")
            if cad==2:
                numeros.append("+")
            if cad==3:
                numeros.append("-")
            if cad==4:
                numeros.append("[")
            if cad==5:
                numeros.append("]")
        return numeros
        
        

class LSystem:
    def __init__(self, axioma, reglas_produccion, iteraciones, angulo, size):
        self.posiciones = []
        self.ventana = turtle.Screen()
        self.ventana.bgcolor("white")
        self.tortuga = turtle.Turtle()
        
        self.axioma = axioma
        self.reglas_produccion = reglas_produccion
        self.iteraciones = iteraciones
        self.angulo = angulo
        self.size = size
        
        # Configuración inicial de la tortuga
        self.tortuga.color("black")
        self.tortuga.speed(0)
        self.tortuga.left(90)
        self.tortuga.penup()
        self.tortuga.setpos(0, -200)
        self.tortuga.pendown()

    def limpiar_dibujo(self):
        """Limpia el lienzo para evitar superposiciones."""
        self.tortuga.clear()
        self.posiciones = []

    def guardar_estado(self):
        self.tortuga.color("green")
        self.tortuga.begin_fill()

        tamaño_hoja = 5
        self.tortuga.circle(tamaño_hoja)
        self.tortuga.end_fill()

        self.tortuga.color("pink")
        self.tortuga.begin_fill()
        self.tortuga.circle(tamaño_hoja / 2)
        self.tortuga.end_fill()
        self.tortuga.color("black")

        self.posiciones.append((self.tortuga.pos(), self.tortuga.heading()))

    def recuperar_estado(self):
        if self.posiciones:
            posicion, direccion = self.posiciones.pop()
            self.tortuga.penup()
            self.tortuga.setpos(posicion)
            self.tortuga.setheading(direccion)
            self.tortuga.pendown()

    def interpretar_cadena(self, cadena):
        for simbolo in cadena:
            self.ventana.tracer(0)
            if simbolo == 'F' or simbolo == 'G':
                self.tortuga.pensize(1)
                self.tortuga.forward(self.size)
            elif simbolo == '+':
                self.tortuga.right(self.angulo)
            elif simbolo == '-':
                self.tortuga.left(self.angulo)
            elif simbolo == '[':
                self.guardar_estado()
            elif simbolo == ']':
                self.recuperar_estado()
        self.ventana.update()

    def aplicar_reglas(self):
        cadena = self.axioma
        for _ in range(self.iteraciones):
            nueva_cadena = ""
            for simbolo in cadena:
                if simbolo in self.reglas_produccion:
                    nueva_cadena += self.reglas_produccion[simbolo]
                else:
                    nueva_cadena += simbolo
            cadena = nueva_cadena
        return cadena

    def guardar_como_imagen(self, folder_path, filename="dibujo"):
        """Guarda la imagen en la carpeta especificada."""
        # Asegurarse de que la carpeta exista
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Crear la carpeta si no existe

        canvas = self.tortuga.getscreen().getcanvas()
        temp_file = f"{folder_path}/{filename}.eps"
        canvas.postscript(file=temp_file)
        try:
            img = Image.open(temp_file)
            png_file = f"{folder_path}/{filename}.png"
            img.save(png_file)
            print(f"Imagen guardada como {png_file}")
            return png_file  # Devuelve la ruta del archivo guardado
        except Exception as e:
            print(f"Error al guardar la imagen: {e}")
            return None

    def generar_y_dibujar(self, folder_path, filename="dibujo"):
        self.limpiar_dibujo()  # Asegúrate de comenzar con un lienzo limpio
        cadena_generada = self.aplicar_reglas()
        self.interpretar_cadena(cadena_generada)
        ruta_imagen = self.guardar_como_imagen(folder_path, filename)
        self.limpiar_dibujo()  # Borra el dibujo actual después de guardar
        return ruta_imagen

        
if __name__ == "__main__":
    
    #ar = Arbol() 
    axioma = 'G'
    iteraciones = 5
    angulo = -22.5
    size = 5
    ag = AlgoritmoGenetico(
        tamano_poblacion=500,
        tamano_cromosoma=15,
        tasa_cruce=0.7,
        tasa_mutacion=0.7,
        num_elitismo=20,
        iteraciones=iteraciones,
        axioma=axioma# Número de individuos élite
    )
    rutas_imagenes=[]
    lista_Reglas =ag.ejecutar(generaciones=100)

    folder_path = "imagenes_generadas"
    for i, regla in enumerate(lista_Reglas, start=1):
        reglas_produccion = {'F': 'FF', 'G': regla}
        lsystem = LSystem(axioma, reglas_produccion, iteraciones, angulo, size)
        ruta = lsystem.generar_y_dibujar(folder_path, filename=f"dibujo_{i}")
        if ruta:
            rutas_imagenes.append(ruta)

    imagenes = [Image.open(ruta) for ruta in rutas_imagenes]

    # Crear el GIF, estableciendo el primer parámetro como la imagen de inicio
    # y los siguientes parámetros como las imágenes a agregar al GIF.
    # La opción "duration" controla el tiempo entre cada fotograma (en milisegundos)
    # "loop" define cuántas veces se repite el GIF, 0 significa infinito
    imagenes[0].save("animacion2.gif", save_all=True, append_images=imagenes[1:], duration=500, loop=0)

    print("GIF creado exitosamente.")
    # Finalizar ventana
    turtle.done()
