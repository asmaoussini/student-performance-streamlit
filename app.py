import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import sklearn
import numpy as np
# --- Titre ---
st.title("📊 Analyse des Performances des Étudiants")
st.markdown("Exploration, Visualisation, et Prédiction des résultats en mathématiques.")




# --- Chargement des données ---
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

# --- Affichage brut ---
if st.checkbox("Afficher les données brutes"):
    st.dataframe(df)

# --- Nettoyage / Encodage ---
df_encoded = df.copy()
df_encoded["gender"] = df["gender"].map({"female": 0, "male": 1})
df_encoded["lunch"] = df["lunch"].map({"standard": 1, "free/reduced": 0})
df_encoded["test preparation course"] = df["test preparation course"].map({"none": 0, "completed": 1})
df_encoded["race/ethnicity"] = df["race/ethnicity"].astype("category").cat.codes
df_encoded["parental level of education"] = df["parental level of education"].astype("category").cat.codes

# --- Visualisation ---
st.subheader("🎨 Visualisations")

# Distribution des notes
fig1, ax1 = plt.subplots()
sns.histplot(df["math score"], bins=20, kde=True, color="skyblue", ax=ax1)
ax1.set_title("Distribution des scores en mathématiques")
st.pyplot(fig1)

# Boxplot des scores selon le genre
fig2, ax2 = plt.subplots()
sns.boxplot(x="gender", y="math score", data=df, palette="Set2", ax=ax2)
ax2.set_xticklabels(["Femme", "Homme"])
ax2.set_title("Score en mathématiques selon le genre")
st.pyplot(fig2)

# --- Modélisation Machine Learning ---
st.subheader("📈 Prédiction des scores en mathématiques")

# Choix des variables
features = ["reading score", "writing score", "gender", "lunch", "test preparation course"]
X = df_encoded[features]
y = df_encoded["math score"]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de régression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.markdown(f"**R² score :** {r2_score(y_test, y_pred):.2f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Calcul manuel du RMSE
st.markdown(f"**RMSE :** {rmse:.2f}")

# --- Téléchargement ---
st.subheader("⬇️ Télécharger les données traitées")
csv = df_encoded.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Télécharger le CSV",
    data=csv,
    file_name='donnees_etudiantes_traitees.csv',
    mime='text/csv'
)
