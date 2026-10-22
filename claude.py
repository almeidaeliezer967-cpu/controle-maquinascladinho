"""
Dashboard de Manutenção Preditiva
---------------------------------
Detecção de anomalias em sensores de chão de fábrica com DBSCAN.

Executar:
    pip install streamlit numpy pandas scikit-learn seaborn matplotlib
    streamlit run dashboard_manutencao_preditiva.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuração da página
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Manutenção Preditiva | DBSCAN",
    page_icon="🏭",
    layout="wide",
)

COL_VIB = "Frequencia_Vibracao"
COL_TEMP = "Pico_Temperatura"
LABEL_ANOMALIA = "Anomalia (outlier)"


# --------------------------------------------------------------------------
# Geração dos dados (NumPy)
# --------------------------------------------------------------------------
@st.cache_data
def gerar_dados(n_por_regime: int, n_outliers: int, seed: int) -> pd.DataFrame:
    """Gera leituras de sensores com 3 regimes normais de operação + outliers."""
    rng = np.random.default_rng(seed)

    # (média_vibração Hz, desvio, média_temperatura °C, desvio)
    regimes = [
        (50, 3.0, 70, 3.0),    # regime de baixa carga
        (80, 4.0, 85, 4.0),    # regime de carga nominal
        (115, 4.5, 68, 3.0),   # regime de alta rotação / boa refrigeração
    ]

    vib, temp = [], []
    for mu_v, sd_v, mu_t, sd_t in regimes:
        vib.append(rng.normal(mu_v, sd_v, n_por_regime))
        temp.append(rng.normal(mu_t, sd_t, n_por_regime))

    # Outliers intencionais: falhas de rolamento, superaquecimento, sensor com defeito
    out_vib = np.concatenate([
        rng.uniform(150, 200, n_outliers // 3),                       # vibração extrema
        rng.uniform(60, 100, n_outliers // 3),                        # superaquecimento
        rng.uniform(5, 25, n_outliers - 2 * (n_outliers // 3)),       # leitura anômala baixa
    ])
    out_temp = np.concatenate([
        rng.uniform(75, 100, n_outliers // 3),
        rng.uniform(115, 140, n_outliers // 3),
        rng.uniform(30, 45, n_outliers - 2 * (n_outliers // 3)),
    ])

    vib.append(out_vib)
    temp.append(out_temp)

    df = pd.DataFrame({
        COL_VIB: np.concatenate(vib),
        COL_TEMP: np.concatenate(temp),
    }).round(2)

    # Embaralha e atribui IDs de máquina
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "Maquina_ID", [f"MAQ-{i + 1:04d}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------
# DBSCAN
# --------------------------------------------------------------------------
def aplicar_dbscan(df: pd.DataFrame, eps: float, min_samples: int) -> pd.DataFrame:
    """Padroniza as métricas e aplica DBSCAN. Rótulo -1 = ruído (anomalia)."""
    X = StandardScaler().fit_transform(df[[COL_VIB, COL_TEMP]])
    rotulos = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    resultado = df.copy()
    resultado["Cluster_ID"] = rotulos
    resultado["Status"] = np.where(rotulos == -1, "Anômala", "Normal")
    resultado["Grupo"] = np.where(
        rotulos == -1, LABEL_ANOMALIA, [f"Cluster {c}" for c in rotulos]
    )
    return resultado


# --------------------------------------------------------------------------
# Gráfico (Seaborn)
# --------------------------------------------------------------------------
def plotar(resultado: pd.DataFrame) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 6))

    normais = resultado[resultado["Cluster_ID"] != -1]
    anomalas = resultado[resultado["Cluster_ID"] == -1]

    # Clusters coloridos (paleta sem preto)
    if not normais.empty:
        ordem = sorted(normais["Grupo"].unique(), key=lambda g: int(g.split()[-1]))
        paleta = dict(zip(ordem, sns.color_palette("tab10", n_colors=max(len(ordem), 3))))
        sns.scatterplot(
            data=normais, x=COL_VIB, y=COL_TEMP,
            hue="Grupo", hue_order=ordem, palette=paleta,
            s=55, alpha=0.8, edgecolor="white", linewidth=0.5, ax=ax,
        )

    # Outliers destacados em preto
    if not anomalas.empty:
        sns.scatterplot(
            data=anomalas, x=COL_VIB, y=COL_TEMP,
            color="black", marker="X", s=130, label=LABEL_ANOMALIA, ax=ax,
        )

    ax.set_xlabel("Frequência de Vibração (Hz)")
    ax.set_ylabel("Pico de Temperatura (°C)")
    ax.set_title("Clusters de operação normal e anomalias (DBSCAN)", fontsize=14, weight="bold")
    ax.legend(title="Agrupamento", loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
st.title("🏭 Dashboard de Manutenção Preditiva")
st.caption(
    "Detecção não supervisionada de anomalias em sensores de chão de fábrica "
    "usando DBSCAN."
)

# Sidebar: parâmetros
with st.sidebar:
    st.header("⚙️ Parâmetros do DBSCAN")
    eps = st.slider(
        "eps (raio de vizinhança)", 0.05, 1.50, 0.30, 0.01,
        help="Distância máxima entre dois pontos para serem vizinhos "
             "(calculada nos dados padronizados).",
    )
    min_samples = st.slider(
        "min_samples", 2, 40, 8, 1,
        help="Número mínimo de pontos na vizinhança para formar uma região densa.",
    )

    st.divider()
    st.header("🧪 Dados simulados")
    n_por_regime = st.slider("Leituras por regime normal", 50, 500, 150, 10)
    n_outliers = st.slider("Outliers intencionais", 3, 60, 18, 1)
    seed = st.number_input("Semente aleatória", min_value=0, value=42, step=1)

# Pipeline
dados = gerar_dados(n_por_regime, n_outliers, int(seed))
resultado = aplicar_dbscan(dados, eps, min_samples)

anomalas = resultado[resultado["Status"] == "Anômala"]
n_clusters = resultado.loc[resultado["Cluster_ID"] != -1, "Cluster_ID"].nunique()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Máquinas monitoradas", f"{len(resultado)}")
c2.metric("Clusters normais", f"{n_clusters}")
c3.metric("Máquinas anômalas", f"{len(anomalas)}")
c4.metric("Taxa de anomalias", f"{len(anomalas) / len(resultado):.1%}")

if n_clusters == 0:
    st.warning(
        "Nenhum cluster foi formado: todos os pontos foram classificados como ruído. "
        "Aumente **eps** ou diminua **min_samples**."
    )

# Gráfico principal
fig = plotar(resultado)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Tabela interativa de máquinas anômalas
st.subheader("🚨 Máquinas consideradas anômalas")
if anomalas.empty:
    st.success("Nenhuma anomalia detectada com os parâmetros atuais.")
else:
    tabela = (
        anomalas[["Maquina_ID", COL_VIB, COL_TEMP]]
        .sort_values([COL_TEMP, COL_VIB], ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Maquina_ID": st.column_config.TextColumn("Máquina"),
            COL_VIB: st.column_config.NumberColumn("Frequência de Vibração (Hz)", format="%.2f"),
            COL_TEMP: st.column_config.NumberColumn("Pico de Temperatura (°C)", format="%.2f"),
        },
    )
    st.download_button(
        "⬇️ Baixar anomalias (CSV)",
        data=tabela.to_csv(index=False).encode("utf-8"),
        file_name="maquinas_anomalas.csv",
        mime="text/csv",
    )

with st.expander("ℹ️ Como interpretar"):
    st.markdown(
        """
- **Clusters coloridos**: regiões densas de leituras, que representam regimes normais de operação.
- **X pretos**: pontos de ruído do DBSCAN (rótulo `-1`), tratados como anomalias.
- **eps**: quanto maior, mais permissivo o agrupamento (menos anomalias).
- **min_samples**: quanto maior, mais exigente a definição de região densa (mais anomalias).
- As métricas são padronizadas (`StandardScaler`) antes do DBSCAN, para que vibração (Hz) e
  temperatura (°C) tenham o mesmo peso na distância.
        """
    )
