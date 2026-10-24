import joblib
import streamlit as st
import pandas as pd


st.set_page_config(page_title="Prédiction Salaire", page_icon="💰", layout="centered")

# 2 CSS 
st.markdown("""
<style>
  .main { background-color: #f8f9fa; }
  .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px; border-radius: 15px; color: white; text-align: center;
        margin-top: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
  .result-card h2 { color: white; margin: 0; }
  .stButton > button {
        width: 100%; background-color: #1e3c72; color: white;
        border-radius: 10px; height: 50px; font-size: 18px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

model = joblib.load('model_salires.pkl')
colonnes = list(model.feature_names_in_)
niveaux = [c.replace('المستوى_الدراسي_', '') for c in colonnes if 'المستوى_الدراسي_' in c]
st.title("💰 Prédiction de Salaire")
st.markdown("---")
st.write("Remplissez vos informations pour estimer votre salaire mensuel")

col1, col2 = st.columns(2)
with col1:
    khibra = st.number_input("📅 Expérience (années)", min_value=0, max_value=50, value=2)
with col2:
    mostawa = st.selectbox("🎓 Niveau d'étude", niveaux)


if st.button("🚀 Prédire le Salaire"):
    df = pd.DataFrame([[khibra, mostawa]], columns=['الخبرة', 'المستوى_الدراسي'])
    df = pd.get_dummies(df, columns=['المستوى_الدراسي'])
    df = df.reindex(columns=colonnes, fill_value=0)
    salaire = model.predict(df)[0]

    st.toast("Prédiction terminée avec succès!", icon="✅")

    st.markdown(f"""
    <div class="result-card">
        <p style="font-size:16px; margin:0;">Salaire Mensuel Estimé</p>
        <h2>{salaire:,.2f} DH</h2>
        <p style="font-size:14px; margin-top:10px;">Expérience: {khibra} ans | Niveau: {mostawa}</p>
    </div>
    """, unsafe_allow_html=True)

    st.metric(label="Résultat", value=f"{salaire:.2f} DH")

st.markdown("---")
st.caption("© 2026 - Modèle Machine Learning | Prédiction basée sur l'expérience et le niveau d'étude")

