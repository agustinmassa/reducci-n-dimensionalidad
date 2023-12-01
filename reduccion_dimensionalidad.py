"""
REDUCCION DE DIMENSIONALIDAD CON LOS DATASETS AnsurMen.csv y AnsurWomen.csv

a) Carguen los dos csvs en dos dataframes distintos de pandas. 
Agréguenle a cada uno una nueva columna 'SEXO' que tenga los valores 'H' y 'M', 
según corresponda, para poder identificar de qué dataset vino cada persona. 
Luego unan los dos datasets en uno nuevo usando la función de pandas pd.concat([df1, df2]).

b) Definan un nuevo dataframe de variables sólo numéricas a partir del anterior, 
descartando las columnas 'SEXO' y 'SUBJECT_NUMBER' (¿tiene sentido quedarse con esta última columna?). 
Luego apliquenle el StandardScaler de sklearn a este nuevo dataframe, y hagan una reducción dimensional usando PCA. 
¿Con cuántas componentes necesito quedarme para explicar el 95% de la varianza de los datos?

c) Ahora hagan otro PCA, pero quedándose sólo con 2 componentes, y hagan un scatterplot de los datos. 
¿Qué es lo que se ve? Traten de pintar los puntos usando la columna categórica "SEXO" que tiene el dataset original.

d) Ahora hagan un PCA con un número reducido de componentes (digamos 8), y luego apliquen un TSNE con 2 componentes. 
Grafiquen los resultados cómo hicieron en el punto anterior. 
¿Qué se ve ahora? Pueden jugar con el número de componentes del PCA, o sólo hacer TSNE, y ver las diferencias.

"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

#A)

MEN = 'F:\Agu\Diplomatura.UNSAM\Practica\Reduccion de dimensionalidad\AnsurMen.csv'
WOMEN = 'F:\Agu\Diplomatura.UNSAM\Practica\Reduccion de dimensionalidad\AnsurWomen.csv'

hombres = pd.read_csv(MEN)
mujeres = pd.read_csv(WOMEN)

hombres['SEXO'] = 'H'
mujeres['SEXO'] = 'M'

personas = pd.concat([hombres, mujeres])

#print(personas.head())

#B)

personas_num = personas.drop(columns=['SEXO', 'SUBJECT_NUMBER'])

scaler = StandardScaler()
personas_num_scaled = scaler.fit_transform(personas_num)

pca = PCA() 
personas_num_pca = pca.fit_transform(personas_num_scaled)
#Hago un PCA si especificar dimensiones
#luego veo el numero necesario para llegar a 0.95 de varianza

var_frac = 0.95
cumsum = np.cumsum(pca.explained_variance_ratio_)
#Esto nos dice cuanta informacion es retenida si paramos en cada dimension
d = np.argmax(cumsum >= var_frac) + 1
#Aca nos dice en que momento llega a var_frac * 100%
print('Con {} componentes, preservamos el {} de la varianza.' .format(d, var_frac))

plt.figure(figsize=(8,5))
plt.plot(cumsum, linewidth=3)

plt.axvline(d, color="k", ls=":")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")

plt.xlabel('Componentes', fontsize=16)
plt.ylabel('Varianza', fontsize=16)

plt.grid(True)
#plt.show()

#C) 

pca_2d = PCA(n_components=2)
personas_pca2d = pca_2d.fit_transform(personas_num_scaled)

colores = {'H':'g', 'M':'orange'}
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
scat = ax.scatter(*personas_pca2d.T, c=personas['SEXO'].map(colores),
                  edgecolors='None', alpha=0.4)
plt.title('PCA 2d')

#Hago una leyenda
leyenda = []
clase = []
for sexo, color in colores.items():
    clase.append(sexo)
    leyenda.append(mpatches.Rectangle((0,0),1,1,fc=color))
plt.legend(leyenda, clase, loc=4)
#plt.show()

#D)

pca_8d = PCA(n_components=8)
personas_pca8d = pca_8d.fit_transform(personas_num_scaled)

tsne = TSNE(n_components=2, random_state=42)
reduced_tsne = tsne.fit_transform(personas_pca8d)

colores = {'H':'g', 'M':'orange'}
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
scat = ax.scatter(*reduced_tsne.T, c=personas['SEXO'].map(colores),
                  edgecolors='None', alpha=0.4)
plt.title('PCA + tSNE 2d')

leyenda = []
clase = []
for sexo, color in colores.items():
    clase.append(sexo)
    leyenda.append(mpatches.Rectangle((0,0),1,1,fc=color))
plt.legend(leyenda, clase, loc=4)
plt.show()