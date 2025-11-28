# Arquivo: app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuração da Página ---
st.set_page_config(page_title="Sistema Athena - Prevenção à Evasão", layout="wide")

# --- 1. Carregar o Cérebro da IA ---
# Usa cache para não recarregar toda vez que clicar num botão
@st.cache_resource
def load_assets():
    model = joblib.load('modelo_final.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    colunas_info = joblib.load('colunas_info.joblib')
    return model, preprocessor, colunas_info

try:
    model, preprocessor, colunas_info = load_assets()
except FileNotFoundError:
    st.error("Erro: Arquivos do modelo não encontrados. Por favor, execute 'python train_model.py' primeiro.")
    st.stop()

# --- 2. Interface Lateral (Inputs) ---
st.sidebar.title("🛡️ Sistema Athena")
st.sidebar.markdown("### Perfil da Estudante")
st.sidebar.info("Insira os dados atualizados para análise de risco.")

# Inputs baseados nos Fatores Críticos descobertos
tuition_fees = st.sidebar.selectbox(
    "Mensalidades em Dia?",
    options=[1, 0],
    format_func=lambda x: "Sim" if x == 1 else "Não (Atrasado)"
)

# Sliders acadêmicos
st.sidebar.markdown("---")
st.sidebar.markdown("**Desempenho Acadêmico**")
units_approved_1st = st.sidebar.slider("Disciplinas Aprovadas (1º Sem)", 0, 20, 5)
units_approved_2nd = st.sidebar.slider("Disciplinas Aprovadas (2º Sem)", 0, 20, 5)
units_enrolled_2nd = st.sidebar.slider("Disciplinas Matriculadas (2º Sem)", 0, 20, 6)

# Dados Pessoais
st.sidebar.markdown("---")
st.sidebar.markdown("**Dados Pessoais**")
age = st.sidebar.number_input("Idade", min_value=17, max_value=70, value=20)
debtor = st.sidebar.selectbox("Possui Dívidas Extras?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
scholarship = st.sidebar.selectbox("É Bolsista?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

# --- 3. Preparar os Dados para a IA ---
input_data = {}

# Preencher tudo com valores padrão (neutros)
for col in colunas_info['num']:
    input_data[col] = [0] 
for col in colunas_info['cat']:
    input_data[col] = [1] 

# Atualizar com o que o usuário digitou
input_data['Tuition fees up to date'] = [tuition_fees]
input_data['Curricular units 2nd sem (approved)'] = [units_approved_2nd]
input_data['Curricular units 1st sem (approved)'] = [units_approved_1st]
input_data['Curricular units 2nd sem (enrolled)'] = [units_enrolled_2nd]
input_data['Age at enrollment'] = [age]
input_data['Debtor'] = [debtor]
input_data['Scholarship holder'] = [scholarship]

df_input = pd.DataFrame(input_data)

# --- 4. Previsão ---
X_input = preprocessor.transform(df_input)
probability = model.predict_proba(X_input)[0][1] # Probabilidade de ser 1 (Evasão)

# --- 5. Dashboard Visual ---
st.title("Monitoramento de Permanência Feminina")
st.markdown("Análise preditiva para suporte à decisão institucional.")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Resultado da Análise")
    
    # Lógica de Cores e Alertas
    if probability > 0.5: # Risco Alto
        st.error(f"⚠️ **RISCO CRÍTICO DETECTADO**")
        st.metric(label="Probabilidade de Evasão", value=f"{probability*100:.1f}%", delta="Alto Risco")
        
        st.markdown("#### 📢 Plano de Ação Recomendado:")
        if tuition_fees == 0:
            st.warning("👉 **Financeiro:** Estudante inadimplente. Acionar política de refinanciamento/bolsa emergencial.")
        if units_approved_1st < 5 or units_approved_2nd < 5:
            st.warning("👉 **Pedagógico:** Baixo índice de aprovação. Encaminhar para tutoria e reforço.")
        if age > 30:
            st.info("👉 **Apoio Social:** Estudante madura. Verificar conflito de horários trabalho/estudo.")
            
    else: # Risco Baixo
        st.success(f"✅ **SITUAÇÃO ESTÁVEL**")
        st.metric(label="Probabilidade de Evasão", value=f"{probability*100:.1f}%", delta="Seguro")
        st.markdown("A estudante apresenta bons indicadores de permanência. Manter acompanhamento regular.")

with col2:
    st.subheader("Indicadores Chave")
    # Barras visuais
    chart_data = pd.DataFrame({
        'Fator': ['Financeiro (Inadimplência)', 'Acadêmico (Baixa Aprov.)', 'Dívidas Extras'],
        'Risco': [1 if tuition_fees==0 else 0, 
                  1 if units_approved_2nd < 3 else 0,
                  1 if debtor==1 else 0]
    })
    st.bar_chart(chart_data.set_index('Fator'), color="#ff4b4b")
    st.caption("Barras cheias indicam presença de fator de risco.")