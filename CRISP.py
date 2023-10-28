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

#%% FASE IV. MODELADO DE DATOS
#%%% Train & Test split

df = pd.read_csv("file.csv")

X = df.drop("target", axis=1)  # Características
y = df["target"]  # Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#%%% Regresión Logística

model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir las etiquetas en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)


#%%% Random Forest

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Puedes ajustar los hiperparámetros según sea necesario
rf_model.fit(X_train, y_train)

# Predecir las etiquetas en el conjunto de prueba
y_pred_rf = rf_model.predict(X_test)

# Calcular métricas de evaluación
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)


#%%% Naive Bayes

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predecir las etiquetas en el conjunto de prueba
y_pred_nb = nb_model.predict(X_test)

# Calcular métricas de evaluación
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)


#%% FASE V. EVALUACIÓN

#%%% Métricas de evaluación
#     Regresión logística
print('\nLogistic Regression\n')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'ROC AUC: {roc_auc}')


#       Bosque Aleatorio
print('\nRandom Forest\n')
print(f' Accuracy: {accuracy_rf}')
print(f'Precision: {precision_rf}')
print(f'Recall: {recall_rf}')
print(f'F1-Score: {f1_rf}')
print(f'Confusion Matrix:\n{conf_matrix_rf}')
print(f'ROC AUC: {roc_auc_rf}')


#       Naive Bayes
print('\nNaive Bayes\n')
print(f'Accuracy: {accuracy_nb}')
print(f'Precision: {precision_nb}')
print(f'Recall: {recall_nb}')
print(f'F1-Score: {f1_nb}')
print(f'Confusion Matrix:\n{conf_matrix_nb}')


#%%% Gráficas ROC

#     Regresión logística
y_prob = model.predict_proba(X_test)[:,1]  # Probabilidad de la clase positiva
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression - Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#       Bosque Aleatorio
y_prob_rf = rf_model.predict_proba(X_test)[:,1]  # Probabilidad de la clase positiva
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'AUC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest - Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


#       Naive Bayes
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, y_prob_nb)
roc_auc_nb = roc_auc_score(y_test, y_prob_nb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, label=f'AUC = {roc_auc_nb:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multinomial Naive Bayes - Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show

#%%% Exportar modelo
import joblib

joblib.dump(rf_model, 'modelo.pkl')
