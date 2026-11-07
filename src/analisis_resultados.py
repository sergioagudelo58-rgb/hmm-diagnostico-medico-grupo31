"""
Análisis estadístico de resultados del HMM (diagnóstico médico).
Lee resultados/simulaciones.csv y genera:
  - resultados/estadisticas_por_escenario.csv  (media, DE, mín, máx, IC 95 %)
  - resultados/pruebas_estadisticas.csv        (Kruskal-Wallis y Welch por pares)
  - figuras/fig1_exactitud_boxplot.png         (300 dpi)
  - figuras/fig3_logverosimilitud_boxplot.png  (300 dpi)
  - figuras/fig2_reales_vs_viterbi.png         (300 dpi, si existe secuencia_ejemplo.csv)
Ejecutar después de src/experimentos.py.

Uso (desde la carpeta raíz del proyecto):
    python src/analisis_resultados.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Rutas relativas a la raíz del proyecto (funciona desde cualquier carpeta)
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DIR_RES = os.path.join(RAIZ, "resultados")
DIR_FIG = os.path.join(RAIZ, "figuras")
RUTA_SIM = os.path.join(DIR_RES, "simulaciones.csv")
RUTA_SEQ = os.path.join(DIR_RES, "secuencia_ejemplo.csv")
os.makedirs(DIR_RES, exist_ok=True)
os.makedirs(DIR_FIG, exist_ok=True)

ESC = ["E1_Base", "E2_Riesgo", "E3_Ruidoso"]
ETIQ = {"E1_Base": "E1 Base", "E2_Riesgo": "E2 Paciente de riesgo",
        "E3_Ruidoso": "E3 Observación ruidosa"}
COLOR = {"E1_Base": "#2a6f97", "E2_Riesgo": "#e07a1f", "E3_Ruidoso": "#8a8a8a"}
METRICAS = ["exactitud", "log_verosimilitud", "error_A", "error_B"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(RUTA_SIM)

# 1. Estadísticas descriptivas con IC 95 % (t de Student)
filas = []
for esc in ESC:
    for m in METRICAS:
        x = df.loc[df.escenario == esc, m].to_numpy()
        n, media, de = len(x), x.mean(), x.std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * de / np.sqrt(n)
        filas.append(dict(escenario=esc, metrica=m, n=n, media=media, de=de,
                          minimo=x.min(), maximo=x.max(),
                          ic95_inf=media - h, ic95_sup=media + h))
est = pd.DataFrame(filas)
est.to_csv(f"{DIR_RES}/estadisticas_por_escenario.csv", index=False)

# 2. Pruebas de comparación entre escenarios
pruebas = []
for m in METRICAS:
    grupos = [df.loc[df.escenario == e, m] for e in ESC]
    H, p = stats.kruskal(*grupos)
    pruebas.append(dict(metrica=m, comparacion="Kruskal-Wallis (3 escenarios)",
                        estadistico=H, p_valor=p, d_cohen=np.nan))
    for a, b in [("E1_Base", "E2_Riesgo"), ("E1_Base", "E3_Ruidoso"),
                 ("E2_Riesgo", "E3_Ruidoso")]:
        xa, xb = df.loc[df.escenario == a, m], df.loc[df.escenario == b, m]
        t, p = stats.ttest_ind(xa, xb, equal_var=False)
        sp = np.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2)
        pruebas.append(dict(metrica=m, comparacion=f"Welch {a} vs {b}",
                            estadistico=t, p_valor=min(p * 3, 1.0),  # Bonferroni
                            d_cohen=(xa.mean() - xb.mean()) / sp))
pd.DataFrame(pruebas).to_csv(f"{DIR_RES}/pruebas_estadisticas.csv", index=False)


def boxplot(metrica, ylabel, titulo, archivo, loc_leyenda="lower left"):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    datos = [df.loc[df.escenario == e, metrica] for e in ESC]
    bp = ax.boxplot(datos, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.4),
                    showfliers=False)
    for patch, e in zip(bp["boxes"], ESC):
        patch.set_facecolor(COLOR[e]); patch.set_alpha(0.55)
    rng = np.random.default_rng(0)  # solo para dispersar puntos (jitter)
    for i, (e, d) in enumerate(zip(ESC, datos), start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=8,
                   color=COLOR[e], alpha=0.5, zorder=3)
        ax.scatter(i, d.mean(), marker="D", s=36, color="white",
                   edgecolor="black", zorder=4)
    ax.set_xticks(range(1, 4), [ETIQ[e] for e in ESC])
    ax.set_xlabel("Escenario de simulación")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontsize=11)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="black", lw=1.4, label="Mediana"),
                       Line2D([], [], marker="D", ls="", mfc="white",
                              mec="black", label="Media"),
                       Line2D([], [], marker="o", ls="", color="gray",
                              alpha=0.6, label="Réplica (n = 100)")],
              loc=loc_leyenda, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{DIR_FIG}/{archivo}", dpi=300)
    plt.close(fig)


boxplot("exactitud", "Exactitud de Viterbi (proporción de días correctos)",
        "Exactitud de la decodificación de Viterbi por escenario",
        "fig1_exactitud_boxplot.png")
boxplot("log_verosimilitud", "Log-verosimilitud log P(O | λ)",
        "Log-verosimilitud de las secuencias observadas por escenario",
        "fig3_logverosimilitud_boxplot.png", loc_leyenda="upper right")

# Figura 2: estados reales vs. Viterbi (requiere secuencia_ejemplo.csv)
if os.path.exists(RUTA_SEQ):
    seq = pd.read_csv(RUTA_SEQ)
    cols = {c.lower(): c for c in seq.columns}
    c_real = next(cols[c] for c in cols if "real" in c)
    c_vit = next(cols[c] for c in cols if "viterbi" in c or "inferid" in c)
    nombres = ["Sano", "Gripe", "Neumonía"]
    def a_indice(s):
        if pd.api.types.is_numeric_dtype(s):
            return s.to_numpy().astype(int)
        mapa = {"sano": 0, "gripe": 1, "neumonia": 2, "neumonía": 2}
        return s.str.lower().map(mapa).to_numpy()
    real, vit = a_indice(seq[c_real]), a_indice(seq[c_vit])
    t = np.arange(1, len(real) + 1)
    acc = (real == vit).mean()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.step(t, real, where="mid", color="#2a6f97", lw=2.2, label="Estado real")
    ax.step(t, vit, where="mid", color="#e07a1f", lw=1.4, ls="--",
            label="Estado inferido (Viterbi)")
    err = real != vit
    ax.scatter(t[err], vit[err], color="#c0392b", s=14, zorder=4,
               label="Día mal clasificado")
    ax.set_yticks([0, 1, 2], nombres)
    ax.set_ylim(-0.4, 2.4)
    ax.set_xlabel("Día (t)")
    ax.set_ylabel("Estado de salud")
    ax.set_title(f"Estados reales vs. decodificación de Viterbi "
                 f"(E1, réplica 1, exactitud = {acc:.2f})", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{DIR_FIG}/fig2_reales_vs_viterbi.png", dpi=300)
    plt.close(fig)
else:
    print(f"Aviso: no se encontró {RUTA_SEQ}; se omite la Figura 2.")

pd.set_option("display.width", 200)
print(est.round(4).to_string(index=False))
print(pd.DataFrame(pruebas).round(5).to_string(index=False))
