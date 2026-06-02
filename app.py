import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuração da Página com Paleta Lilás/Rosa ---
st.set_page_config(
    page_title="Sistema Athena - Inteligência e Permanência Feminina", 
    layout="wide", 
    page_icon="🛡️"
)

# --- Customização de Cores via CSS (Lilás e Rosa) ---
st.markdown("""
    <style>
    /* Mudando a barra lateral para um tom lilás bem suave */
    [data-testid="stSidebar"] {
        background-color: #F3EBF6;
    }
    /* Estilizando os títulos e textos em destaque */
    .athena-title {
        color: #6B5B95;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
    }
    .athena-subtitle {
        color: #FF6F61;
        font-size: 1.2rem;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. Carregar o Cérebro da IA ---
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
st.sidebar.markdown("<h2 style='color: #6B5B95;'>🛡️ Sistema Athena</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #8E44AD; font-weight: bold;'>Guardiã da Trajetória Feminina em STEM</p>", unsafe_allow_html=True)
st.sidebar.info("Insira os dados atualizados para análise preditiva de risco.")

# Inputs baseados nos Fatores Críticos descobertos
st.sidebar.markdown("---")
st.sidebar.markdown("<b style='color: #6B5B95;'>Situação Financeira</b>", unsafe_allow_html=True)
tuition_fees = st.sidebar.selectbox(
    "Mensalidades em Dia?",
    options=[1, 0],
    format_func=lambda x: "Sim" if x == 1 else "Não (Atrasado)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<b style='color: #6B5B95;'>Desempenho Acadêmico</b>", unsafe_allow_html=True)
units_approved_1st = st.sidebar.slider("Disciplinas Aprovadas (1º Sem)", 0, 20, 5)
units_approved_2nd = st.sidebar.slider("Disciplinas Aprovadas (2º Sem)", 0, 20, 5)
units_enrolled_2nd = st.sidebar.slider("Disciplinas Matriculadas (2º Sem)", 0, 20, 6)

st.sidebar.markdown("---")
st.sidebar.markdown("<b style='color: #6B5B95;'>Dados Pessoais e Apoio</b>", unsafe_allow_html=True)
age = st.sidebar.number_input("Idade", min_value=17, max_value=70, value=20)
debtor = st.sidebar.selectbox("Possui Dívidas Extras?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
scholarship = st.sidebar.selectbox("É Bolsista?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

# --- 3. Preparar os Dados para a IA ---
input_data = {}

for col in colunas_info['num']:
    input_data[col] = [0] 
for col in colunas_info['cat']:
    input_data[col] = [1] 

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
probability = model.predict_proba(X_input)[0][1] 

# --- 5. Dashboard Visual (Cabeçalho Athena) ---
st.markdown("<h1 class='athena-title'>🛡️ Escudo de Athena: Proteção e Permanência</h1>", unsafe_allow_html=True)
st.markdown("<p class='athena-subtitle'>Plataforma inteligente de monitoramento e salvaguarda de trajetórias femininas na tecnologia.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Resultado da Análise de Vulnerabilidade")
    
    # Lógica de Alertas e Mensagens Customizadas
    if probability > 0.5: 
        st.markdown("<div style='padding:15px; border-radius:10px; background-color:#FFECEA; border-left: 6px solid #FF6F61; color:#C0392B;'>⚠️ <b>ALERTA DO ESCUDO:</b> Risco de vulnerabilidade detectado na permanência da estudante.</div>", unsafe_allow_html=True)
        st.write("")
        st.metric(label="Índice de Risco Estimado pela IA", value=f"{probability*100:.1f}%", delta="Atenção Necessária", delta_color="inverse")
        
        st.markdown("<h4 style='color: #6B5B95;'>📢 Plano de Ação Estratégico (Medidas Protetivas):</h4>", unsafe_allow_html=True)
        if tuition_fees == 0:
            st.warning("👉 **Suporte Financeiro:** Mensalidade em atraso. Acionar com urgência a rede de bolsas emergenciais ou refinanciamento ético.")
        if units_approved_1st < 5 or units_approved_2nd < 5:
            st.warning("👉 **Apoio Pedagógico Coletivo:** Baixo índice de aprovação. Integrar a estudante em grupos de mentoria e apoio técnico acadêmico.")
        if age > 30:
            st.info("👉 **Conciliação e Cuidado:** Aluna madura. Avaliar a flexibilização de prazos e apoio para conciliação entre trabalho/família e faculdade.")
            
    else: 
        st.markdown("<div style='padding:15px; border-radius:10px; background-color:#EAF2F8; border-left: 6px solid #8E44AD; color:#2980B9;'>✅ <b>ESCUDO ATIVO:</b> Indicadores de permanência seguros e estáveis.</div>", unsafe_allow_html=True)
        st.write("")
        st.metric(label="Índice de Risco Estimado pela IA", value=f"{probability*100:.1f}%", delta="Seguro", delta_color="normal")
        st.markdown("<p style='color: #8E44AD;'>A estudante apresenta excelente engajamento. O plano atual é manter o ecossistema protetivo e o acompanhamento regular.</p>", unsafe_allow_html=True)

with col2:
    st.subheader("Fatores de Impacto")
    
    chart_data = pd.DataFrame({
        'Fator Monitorado': ['Financeiro', 'Desempenho Acadêmico', 'Compromissos Externos'],
        'Nível de Alerta': [1 if tuition_fees==0 else 0.1, 
                            1 if units_approved_2nd < 5 else 0.1,
                            1 if debtor==1 else 0.1]
    })
    
    # Gráfico de Barras utilizando a cor Rosa Coral da paleta Athena
    st.bar_chart(chart_data.set_index('Fator Monitorado'), color="#FF6F61")
    st.caption("Barras elevadas destacam componentes específicos que necessitam de intervenção protetiva.")
