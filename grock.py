import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard de Manutenção Preditiva")
st.markdown("""
Este dashboard utiliza **DBSCAN** para identificar clusters de funcionamento normal e anomalias 
em sensores de chão de fábrica. Ajuste os parâmetros abaixo para explorar diferentes configurações de detecção.
""")

# Sidebar com controles
st.sidebar.header("Parâmetros do DBSCAN")
eps = st.sidebar.slider(
    "eps (raio de vizinhança)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.05,
    help="Distância máxima entre dois pontos para serem considerados vizinhos."
)
min_samples = st.sidebar.slider(
    "min_samples (mínimo de pontos)",
    min_value=3,
    max_value=20,
    value=5,
    step=1,
    help="Número mínimo de pontos em uma vizinhança para formar um cluster."
)

# Geração de dados sintéticos
@st.cache_data
def gerar_dados(n_samples=250, random_state=42):
    np.random.seed(random_state)
    
    # Cluster principal de funcionamento normal
    n_normal = int(n_samples * 0.85)
    freq_normal = np.random.normal(loc=45.0, scale=4.5, size=n_normal)
    temp_normal = np.random.normal(loc=65.0, scale=5.0, size=n_normal)
    
    # Segundo cluster de operação em regime diferente (ainda normal)
    n_normal2 = int(n_samples * 0.08)
    freq_normal2 = np.random.normal(loc=55.0, scale=3.0, size=n_normal2)
    temp_normal2 = np.random.normal(loc=72.0, scale=3.5, size=n_normal2)
    
    # Outliers intencionais (anomalias)
    n_outliers = n_samples - n_normal - n_normal2
    freq_out = np.concatenate([
        np.random.uniform(70, 95, size=n_outliers // 2),
        np.random.uniform(10, 25, size=n_outliers - n_outliers // 2)
    ])
    temp_out = np.concatenate([
        np.random.uniform(90, 120, size=n_outliers // 2),
        np.random.uniform(20, 40, size=n_outliers - n_outliers // 2)
    ])
    
    frequencia = np.concatenate([freq_normal, freq_normal2, freq_out])
    temperatura = np.concatenate([temp_normal, temp_normal2, temp_out])
    
    # IDs de máquinas
    maquinas = [f"Maq_{i+1:03d}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        "Maquina": maquinas,
        "Frequencia_Vibracao": frequencia,
        "Pico_Temperatura": temperatura
    })
    return df

df = gerar_dados()

# Preparação dos dados para o modelo
X = df[["Frequencia_Vibracao", "Pico_Temperatura"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicação do DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

df["Cluster"] = labels
df["Status"] = df["Cluster"].apply(lambda x: "Anomalia" if x == -1 else f"Cluster {x}")

# Métricas resumidas
n_anomalias = (labels == -1).sum()
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Máquinas", len(df))
col2.metric("Clusters Detectados", n_clusters)
col3.metric("Anomalias", n_anomalias, delta=f"{n_anomalias/len(df)*100:.1f}%")
col4.metric("eps / min_samples", f"{eps:.2f} / {min_samples}")

# Gráfico de dispersão
st.subheader("Gráfico de Dispersão — Clusters e Anomalias")

fig, ax = plt.subplots(figsize=(10, 6))

# Separar pontos normais e outliers para estilização
mask_normal = labels != -1
mask_outlier = labels == -1

# Plot dos clusters normais
if mask_normal.any():
    sns.scatterplot(
        data=df[mask_normal],
        x="Frequencia_Vibracao",
        y="Pico_Temperatura",
        hue="Cluster",
        palette="viridis",
        s=80,
        alpha=0.8,
        ax=ax,
        legend="full"
    )

# Plot dos outliers em preto
if mask_outlier.any():
    ax.scatter(
        df.loc[mask_outlier, "Frequencia_Vibracao"],
        df.loc[mask_outlier, "Pico_Temperatura"],
        c="black",
        s=120,
        marker="X",
        label="Anomalia (Outlier)",
        zorder=5,
        edgecolors="white",
        linewidths=0.8
    )

ax.set_xlabel("Frequência de Vibração (Hz)", fontsize=12)
ax.set_ylabel("Pico de Temperatura (°C)", fontsize=12)
ax.set_title("Detecção de Anomalias com DBSCAN", fontsize=14, fontweight="bold")
ax.legend(title="Agrupamento", loc="best", frameon=True)
ax.grid(True, linestyle="--", alpha=0.4)
sns.despine()

st.pyplot(fig)

# Tabela de anomalias
st.subheader("Máquinas Consideradas Anômalas")

df_anomalias = df[df["Cluster"] == -1][
    ["Maquina", "Frequencia_Vibracao", "Pico_Temperatura"]
].copy()
df_anomalias["Frequencia_Vibracao"] = df_anomalias["Frequencia_Vibracao"].round(2)
df_anomalias["Pico_Temperatura"] = df_anomalias["Pico_Temperatura"].round(2)
df_anomalias = df_anomalias.reset_index(drop=True)

if len(df_anomalias) > 0:
    st.dataframe(
        df_anomalias,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Maquina": st.column_config.TextColumn("Máquina"),
            "Frequencia_Vibracao": st.column_config.NumberColumn(
                "Frequência de Vibração (Hz)", format="%.2f"
            ),
            "Pico_Temperatura": st.column_config.NumberColumn(
                "Pico de Temperatura (°C)", format="%.2f"
            ),
        }
    )
    st.caption(f"Total de {len(df_anomalias)} máquina(s) identificada(s) como anômala(s).")
else:
    st.info("Nenhuma anomalia detectada com os parâmetros atuais. Tente reduzir o valor de `eps` ou aumentar `min_samples`.")

# Informações adicionais
with st.expander("Sobre o método e os dados"):
    st.markdown("""
    **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**  
    - Identifica clusters de densidade variável e marca pontos isolados como ruído (anomalias).  
    - `eps`: raio máximo de vizinhança.  
    - `min_samples`: quantidade mínima de pontos para formar um núcleo de cluster.  

    **Dados sintéticos gerados:**  
    - Dois grupos de operação normal (frequência ~45 Hz e ~55 Hz).  
    - Outliers intencionais com vibração e temperatura extremas.  
    - Escala padronizada (StandardScaler) antes do DBSCAN para equidade entre as métricas.
    """)

st.markdown("---")
st.caption("Dashboard de Manutenção Preditiva • Python + Streamlit + scikit-learn + Seaborn")
