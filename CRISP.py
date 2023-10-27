"""
Created on Wed Oct 25 00:21:42 2023

@author: Bren Guzmán, María José Merino

"""


#%% LIBRERÍAS

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#%% FASE II. COMPRENSIÓN DE LOS DATOS

#%%% Fuente de datos
# Recopilación de los datos
df = pd.read_csv("data.csv")

#%%% Exploración de los datos
#%%%% Información de las variables
class_distribution = df['satisfaction_v2'].value_counts() #verificar si está balanceado
print(class_distribution)

df.drop('id', axis=1, inplace=True) #eliminamo la columna de índice
print("\n\nDescribe: \n",df.describe()) #estadísticos básicos
print("\n\n NaN Values: \n",df.isna().sum()) #Valores nulos
print("\n\nInfo:\n",df.info) #Información de dataframe
print("\n\nTipos:\n",df.dtypes) #Tipos de datos
print("\n\nValores únicos:\n",df.nunique()) #valores únicos

#%%%% Histogramas

# Se seleccionan las columnas numéricas.
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=10)  
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

#%%%% Countplots
# Se seleccionan las columnas categóricas
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=col)
    plt.title(f'Gráfico de Conteo para {col}')
    plt.xlabel(col)
    plt.ylabel('Conteo')
    plt.xticks(rotation=90)  
    plt.show()

#%%%% Gráficas de caja
for col in num_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Diagrama de Caja de {col}')
    plt.show()
    

#%% FASE III. PREPARACIÓN DE LOS DATOS

#%%% Imputación de valores faltantes

# Calcula la mediana de la columna 'Arrival Delay in Minutes'.
mediana = df['Arrival Delay in Minutes'].median()

# Imputa los valores faltantes con la mediana.
df['Arrival Delay in Minutes'].fillna(mediana, inplace=True)
#%%% Outliers

lower_percentile = 0.05  # Percentil inferior para considerar como valor atípico
upper_percentile = 0.95  # Percentil superior para considerar como valor atípico

# Filtra el DataFrame para eliminar valores atípicos extremos
filtered_df = df.copy()  # Crea una copia del DataFrame original

for col in num_cols:
    lower_bound = df[col].quantile(lower_percentile)
    upper_bound = df[col].quantile(upper_percentile)
    
    # Filtra los valores dentro de los límites de percentiles
    filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]


#%%% Codificación de variables categóricas

#   Columna objetivo

# Crea una instancia de LabelEncoder.
label_encoder = LabelEncoder()

# Ajustamos el LabelEncoder a la columna 'satisfaction_v2' y transforma los valores.
filtered_df['target'] = label_encoder.fit_transform(filtered_df['satisfaction_v2'])
filtered_df.drop('satisfaction_v2', axis=1, inplace=True) #eliminamos la columna satisfaction_v2

#   Otras columnas categóricas

cat_cols = filtered_df.select_dtypes(include=['object']).columns

label_mappings_by_column = {}

for col in cat_cols:
    unique_values = filtered_df[col].unique()
    label_mappings_by_column[col] = {original: encoded for encoded, original in enumerate(unique_values)}

print(label_mappings_by_column)

for col in cat_cols:
    filtered_df[col] = label_encoder.fit_transform(filtered_df[col])
    

#%%% Normalización

# Age, Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes

columns_to_normalize = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

scaler = MinMaxScaler()
filtered_df[columns_to_normalize] = scaler.fit_transform(filtered_df[columns_to_normalize])

#%%% Correlación

# Calcula la matriz de correlación
correlation_matrix = filtered_df.corr()

# Crea una figura de tamaño adecuado
plt.figure(figsize=(10, 8))

# Crea una gráfica de mapa de calor (heatmap) de la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

# Personaliza el título del gráfico
plt.title("Matriz de Correlación")

# Muestra la gráfica
plt.show()


target_correlations = correlation_matrix["target"]
sorted_correlations = target_correlations.abs().sort_values(ascending=False)

#%%% nuevo archivo

filtered_df = filtered_df[['Inflight entertainment', 'Ease of Online booking', 'Online support', 'On-board service', 'Seat comfort', 'Online boarding', 'Leg room service', 'Customer Type', 'Class', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Inflight wifi service', 'target']]

#filtered_df.to_csv('file.csv', index=False)  
