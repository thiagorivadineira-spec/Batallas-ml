import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Predictor de batallas históricas",
    page_icon="⚔️",
    layout="wide",
)

# ---------------------------------------------------------------
# Configuración: columnas que espera el modelo (igual que en el notebook)
# ---------------------------------------------------------------
NUM_COLS = ["attacker_str", "defender_str", "log_ratio_fuerzas", "anio",
            "att_tank", "def_tank", "att_arty", "def_arty", "att_fly", "def_fly"]
VENTAJA_COLS = ["aeroa", "surpa", "cea", "leada", "trnga", "morala",
                "logsa", "momnta", "intela", "techa", "inita"]
CAT_COLS = ["post1"]
COLUMNAS = NUM_COLS + VENTAJA_COLS + CAT_COLS

NOMBRES_VENTAJAS = {
    "leada": "Liderazgo",
    "inita": "Iniciativa",
    "morala": "Moral",
    "intela": "Inteligencia",
    "cea": "Efectividad de combate",
    "surpa": "Sorpresa",
    "momnta": "Impulso (momentum)",
    "trnga": "Entrenamiento",
    "logsa": "Logística",
    "techa": "Tecnología",
    "aeroa": "Superioridad aérea",
}

POSTURAS = {
    "HD": "Defensa apresurada (HD)",
    "PD": "Defensa preparada (PD)",
    "FD": "Defensa fortificada (FD)",
    "DL": "Acción dilatoria (DL)",
    "WD": "Retirada (WD)",
}

OPCIONES_VENTAJA = {-1: "Defensor", 0: "Ninguno", 1: "Atacante"}


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo.pkl")


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variables derivadas y verifica que estén todas las columnas."""
    df = df.copy()
    if "log_ratio_fuerzas" not in df.columns:
        df["log_ratio_fuerzas"] = np.log(df["attacker_str"] / df["defender_str"])
    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el archivo: {', '.join(faltantes)}")
    df = df[COLUMNAS]
    # Mismos tipos que en el entrenamiento: numéricas como float, categórica como texto
    df[NUM_COLS + VENTAJA_COLS] = df[NUM_COLS + VENTAJA_COLS].astype(float)
    df[CAT_COLS] = df[CAT_COLS].astype(object)
    return df


modelo = cargar_modelo()

# ---------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------
st.title("⚔️ Predictor de batallas históricas")
st.write(
    "Modelo de aprendizaje automático entrenado con más de 650 batallas terrestres "
    "entre 1600 y 1973 (dataset CDB90). Estima la probabilidad de que el "
    "**atacante gane** la batalla."
)

tab_sim, tab_csv, tab_info = st.tabs(
    ["🎯 Simular una batalla", "📂 Cargar archivo CSV", "ℹ️ Sobre el modelo"]
)

# ---------------------------------------------------------------
# Pestaña 1: simulación manual
# ---------------------------------------------------------------
with tab_sim:
    st.subheader("Fuerzas de cada bando")
    col_att, col_def = st.columns(2)

    with col_att:
        st.markdown("#### 🔴 Atacante")
        att_str = st.number_input("Tropas", min_value=1, value=20000, step=1000, key="att_str")
        att_arty = st.number_input("Piezas de artillería", min_value=0, value=20, key="att_arty")
        att_tank = st.number_input("Tanques", min_value=0, value=0, key="att_tank")
        att_fly = st.number_input("Aviones", min_value=0, value=0, key="att_fly")

    with col_def:
        st.markdown("#### 🔵 Defensor")
        def_str = st.number_input("Tropas", min_value=1, value=20000, step=1000, key="def_str")
        def_arty = st.number_input("Piezas de artillería", min_value=0, value=20, key="def_arty")
        def_tank = st.number_input("Tanques", min_value=0, value=0, key="def_tank")
        def_fly = st.number_input("Aviones", min_value=0, value=0, key="def_fly")

    st.subheader("Contexto")
    col_a, col_b = st.columns(2)
    with col_a:
        anio = st.slider("Año de la batalla", 1600, 1973, 1800)
    with col_b:
        post1 = st.selectbox("Postura del defensor", list(POSTURAS), format_func=POSTURAS.get)

    st.subheader("¿Quién tiene ventaja en cada aspecto?")
    ventajas = {}
    cols = st.columns(3)
    for i, (codigo, nombre) in enumerate(NOMBRES_VENTAJAS.items()):
        with cols[i % 3]:
            ventajas[codigo] = st.radio(
                nombre,
                options=list(OPCIONES_VENTAJA),
                format_func=OPCIONES_VENTAJA.get,
                index=1,
                horizontal=True,
                key=codigo,
            )

    if st.button("⚔️ Simular batalla", type="primary", use_container_width=True):
        datos = pd.DataFrame([{
            "attacker_str": att_str, "defender_str": def_str, "anio": anio,
            "att_tank": att_tank, "def_tank": def_tank,
            "att_arty": att_arty, "def_arty": def_arty,
            "att_fly": att_fly, "def_fly": def_fly,
            "post1": post1, **ventajas,
        }])
        prob = modelo.predict_proba(preparar(datos))[0, 1]

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad de victoria del atacante", f"{prob:.0%}")
        with c2:
            st.progress(float(prob))
            if prob >= 0.5:
                st.success("🔴 El modelo predice que **gana el atacante**.")
            else:
                st.error("🔵 El modelo predice que **el atacante no gana** (victoria del defensor o empate).")

# ---------------------------------------------------------------
# Pestaña 2: carga de CSV con varias batallas
# ---------------------------------------------------------------
with tab_csv:
    st.write(
        "Subí un archivo CSV con una batalla por fila. Las ventajas se codifican con "
        "**1** (atacante), **0** (ninguno) o **-1** (defensor). Si falta algún valor, "
        "el modelo lo completa automáticamente."
    )

    plantilla = pd.DataFrame([
        {"name": "Batalla ejemplo 1", "attacker_str": 30000, "defender_str": 20000,
         "anio": 1805, "att_tank": 0, "def_tank": 0, "att_arty": 60, "def_arty": 40,
         "att_fly": 0, "def_fly": 0, "post1": "HD",
         **{c: 0 for c in VENTAJA_COLS}, "leada": 1},
        {"name": "Batalla ejemplo 2", "attacker_str": 50000, "defender_str": 60000,
         "anio": 1943, "att_tank": 300, "def_tank": 400, "att_arty": 500, "def_arty": 700,
         "att_fly": 100, "def_fly": 150, "post1": "FD",
         **{c: 0 for c in VENTAJA_COLS}, "morala": -1},
    ])
    st.download_button(
        "⬇️ Descargar plantilla CSV",
        plantilla.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_batallas.csv",
        mime="text/csv",
    )

    archivo = st.file_uploader("Cargar archivo", type="csv")
    if archivo is not None:
        try:
            nuevos = pd.read_csv(archivo)
            X_nuevos = preparar(nuevos)
            probs = modelo.predict_proba(X_nuevos)[:, 1]

            resultado = nuevos.copy()
            resultado["prob_victoria_atacante"] = probs.round(3)
            resultado["prediccion"] = np.where(probs >= 0.5, "Gana atacante", "No gana atacante")

            st.success(f"Se procesaron {len(resultado)} batallas.")
            st.dataframe(resultado, use_container_width=True)
            st.download_button(
                "⬇️ Descargar resultados",
                resultado.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_batallas.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"No se pudo procesar el archivo: {e}")

# ---------------------------------------------------------------
# Pestaña 3: información del modelo
# ---------------------------------------------------------------
with tab_info:
    st.subheader("Dataset")
    st.write(
        "CDB90 (CAA Database of Battles), elaborado por la Agencia de Análisis de "
        "Conceptos del Ejército de EE.UU. Contiene batallas terrestres entre 1600 y 1973."
    )

    st.subheader("Modelo")
    st.write(
        "Se compararon **Regresión Logística** y **Random Forest**. Se eligió la "
        "Regresión Logística por obtener mejores resultados en el conjunto de prueba "
        "y por ser más simple e interpretable."
    )

    st.subheader("Métricas en el conjunto de prueba (132 batallas)")
    st.table(pd.DataFrame({
        "Accuracy": [0.73, 0.70, 0.60],
        "F1 (macro)": [0.72, 0.68, None],
        "ROC-AUC": [0.825, 0.806, 0.50],
    }, index=["Regresión Logística (elegido)", "Random Forest", "Línea base"]))

    st.subheader("Principales conclusiones")
    st.markdown(
        "- El **liderazgo** es el factor que más influye en la victoria del atacante.\n"
        "- La **moral**, la **inteligencia** y la **iniciativa** también tienen un peso importante.\n"
        "- Tener más tropas ayuda, pero pesa menos que la calidad del mando.\n"
        "- Atacar una **posición fortificada** reduce claramente las chances de ganar."
    )
    st.caption("Las predicciones son estimaciones estadísticas basadas en datos históricos.")
