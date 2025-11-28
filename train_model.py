# Arquivo: train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

print("--- Iniciando Processo de Treinamento ---")

# 1. Carregar Dados
try:
    df = pd.read_csv('dataset.csv')
    print("Dataset carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo 'dataset.csv' não foi encontrado na pasta.")
    exit()

# 2. Filtragem e Definição de Colunas (Baseado no que corrigimos)
GENDER_FEMININO_CODE = 1 # Ajuste conforme seu dataset, mantivemos 1 conforme seus testes
df_feminino = df[df['Gender'] == GENDER_FEMININO_CODE].copy()

# Criar Target Binário
df_feminino['Target_Binario'] = np.where(df_feminino['Target'] == 'Dropout', 1, 0)
Y = df_feminino['Target_Binario']

# Listas Corrigidas (com 'Nacionality' e colunas extras)
COLUNAS_CATEGORICAS = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Nacionality', 
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced',
    'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Scholarship holder', 'International'
]

COLUNAS_NUMERICAS = [
    'Age at enrollment', 'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)', 
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

# Selecionar Features
X = df_feminino[COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS]

# 3. Pré-processamento
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, COLUNAS_NUMERICAS),
        ('cat', categorical_transformer, COLUNAS_CATEGORICAS)
    ])

# 4. Divisão Treino/Teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Aplicar pré-processamento no Treino
X_train_processed = preprocessor.fit_transform(X_train)

# 5. Treinar Modelo (Regressão Logística - O Vencedor)
print("Treinando Regressão Logística...")
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train_processed, Y_train)

# 6. Salvar os arquivos para o App
print("Salvando arquivos .joblib...")
joblib.dump(model, 'modelo_final.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')

colunas_info = {'num': COLUNAS_NUMERICAS, 'cat': COLUNAS_CATEGORICAS}
joblib.dump(colunas_info, 'colunas_info.joblib')

print("--- SUCESSO! Execute agora o arquivo app.py ---")