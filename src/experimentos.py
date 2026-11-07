"""
Ejecuta los 3 escenarios x 100 réplicas y guarda los resultados en CSV.

Uso (desde la carpeta raíz del proyecto):
    python src/experimentos.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hmm import HMM, baum_welch, alinear_estados, ESTADOS, SINTOMAS

# ----------------------------------------------------------------------
# Parámetros del experimento
T = 100             # días por paciente simulado
REPLICAS = 100      # simulaciones por escenario
SEMILLA_BASE = 42   # semilla de cada réplica = 42 + número de réplica
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados")

# ----------------------------------------------------------------------
# Escenario E1 (base): matrices del documento, sección 3
PI_BASE = [0.70, 0.25, 0.05]
A_BASE = [[0.80, 0.17, 0.03],
          [0.30, 0.55, 0.15],
          [0.05, 0.25, 0.70]]
B_BASE = [[0.70, 0.15, 0.05, 0.10],
          [0.10, 0.40, 0.30, 0.20],
          [0.02, 0.28, 0.45, 0.25]]

ESCENARIOS = {
    # E1: paciente con riesgo típico
    "E1_Base": HMM(PI_BASE, A_BASE, B_BASE),
    # E2: paciente de riesgo -> empieza más enfermo y la enfermedad dura más
    "E2_Riesgo": HMM([0.30, 0.40, 0.30],
                     [[0.70, 0.22, 0.08],
                      [0.20, 0.65, 0.15],
                      [0.03, 0.17, 0.80]],
                     B_BASE),
    # E3: observación ruidosa -> los síntomas se parecen en todos los estados
    "E3_Ruidoso": HMM(PI_BASE, A_BASE,
                      [[0.40, 0.25, 0.15, 0.20],
                       [0.25, 0.30, 0.25, 0.20],
                       [0.20, 0.28, 0.30, 0.22]]),
}


def correr():
    os.makedirs(CARPETA, exist_ok=True)
    filas = []
    for nombre, modelo in ESCENARIOS.items():
        for r in range(REPLICAS):
            semilla = SEMILLA_BASE + r
            estados, obs = modelo.simular(T, semilla)

            # 1) Viterbi: qué tan bien adivina el estado real
            inferidos = modelo.viterbi(obs)
            exactitud = (inferidos == estados).mean()

            # 2) Forward: qué tan probables son los síntomas bajo el modelo
            log_v, _, _ = modelo.forward(obs)

            # 3) Baum-Welch: qué tan bien se recuperan A y B solo con síntomas
            estimado = baum_welch(obs, modelo.N, modelo.M, semilla)
            A_est, B_est = alinear_estados(estimado, modelo)
            error_A = np.abs(A_est - modelo.A).mean()
            error_B = np.abs(B_est - modelo.B).mean()

            filas.append({"escenario": nombre, "replica": r + 1, "semilla": semilla,
                          "exactitud": exactitud, "log_verosimilitud": log_v,
                          "error_A": error_A, "error_B": error_B})

            # Guardamos una secuencia de ejemplo (E1, réplica 1) para la figura
            if nombre == "E1_Base" and r == 0:
                pd.DataFrame({"dia": np.arange(1, T + 1),
                              "estado_real": [ESTADOS[i] for i in estados],
                              "sintoma": [SINTOMAS[i] for i in obs],
                              "estado_viterbi": [ESTADOS[i] for i in inferidos]}
                             ).to_csv(os.path.join(CARPETA, "secuencia_ejemplo.csv"),
                                      index=False, encoding="utf-8-sig")
        print(f"{nombre}: {REPLICAS} réplicas listas")

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(CARPETA, "simulaciones.csv"), index=False, encoding="utf-8-sig")

    # Resumen rápido para revisar (la tabla final la hace Gilliam)
    resumen = df.groupby("escenario")[["exactitud", "log_verosimilitud", "error_A", "error_B"]] \
                .agg(["mean", "std", "min", "max"]).round(4)
    print("\nResumen:\n", resumen.to_string())
    return df


if __name__ == "__main__":
    correr()
