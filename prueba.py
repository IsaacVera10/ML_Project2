import numpy as np

def Init_Centroide(data, k):
    # Inicialización K-means++
    
    # Paso 1: Elegir el primer centroide al azar de los datos
    centroids = [data[np.random.choice(len(data))].tolist()]

    for _ in range(1, k):
        # Calcular las distancias al cuadrado de cada punto de datos al centroide más cercano
        distances = np.array([min(np.linalg.norm(p - c) ** 2 for c in centroids) for p in data])
        
        # Elegir el nuevo centroide como el punto de datos más lejano
        new_centroid = data[np.argmax(distances)].tolist()
        
        # Agregar el nuevo centroide a la lista de centroides
        centroids.append(new_centroid)
    
    return centroids

# Ejemplo de uso:
data = np.array([[1,1],[3,3], [5,2], [7,1], [7, 4], [1, 5],[-1,3],[-2,5],[-3,3]])
k = 4
centroids = Init_Centroide(data, k)
print(centroids)