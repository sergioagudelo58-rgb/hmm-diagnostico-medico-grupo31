# HMM para diagnóstico médico — Simulación

Modelo Oculto de Markov (HMM) discreto que simula la evolución diaria del estado de salud de un paciente
(**Sano, Gripe, Neumonía**, estados ocultos) a partir de síntomas observados
(**Sin síntomas, Tos, Fiebre, Fatiga**).

Actividad didáctica 2-m2 — Simulación, IU Digital de Antioquia.
Integrantes: Gilliam Argel Monterroza, Leydi Correa Herrera, Sergio Andrés Agudelo Marulanda, Arley Steven Franco Galeano.

> Modelo académico: los parámetros son supuestos ilustrativos y no deben usarse para diagnóstico real.

## Estructura

```
├── src/
│   ├── hmm.py            # Clase HMM: validar, simular, Forward, Viterbi + Baum-Welch
│   └── experimentos.py   # 3 escenarios x 100 réplicas -> CSV
├── resultados/
│   ├── simulaciones.csv       # una fila por réplica
│   └── secuencia_ejemplo.csv  # estados reales vs. Viterbi (E1, réplica 1)
├── figuras/               # gráficas del análisis
├── notebooks/             # análisis de resultados
└── requirements.txt
```

## Instalación y ejecución

```bash
git clone <URL-del-repo>
cd hmm-diagnostico
pip install -r requirements.txt
python src/experimentos.py
```

En Google Colab: sube las carpetas `src/` y `resultados/`, y ejecuta `!python src/experimentos.py`.

## Parámetros del experimento

| Parámetro | Valor |
|---|---|
| Días por secuencia (T) | 100 |
| Réplicas por escenario | 100 |
| Semilla | 42 + número de réplica (42 a 141) |
| Iteraciones máx. Baum-Welch | 100 (tolerancia 1e-4) |

## Escenarios

| Escenario | Cambio respecto al base |
|---|---|
| E1 Base | π = [0.70, 0.25, 0.05]; A y B de la sección 3 del documento |
| E2 Paciente de riesgo | π = [0.30, 0.40, 0.30]; A = [[0.70, 0.22, 0.08], [0.20, 0.65, 0.15], [0.03, 0.17, 0.80]] |
| E3 Observación ruidosa | B = [[0.40, 0.25, 0.15, 0.20], [0.25, 0.30, 0.25, 0.20], [0.20, 0.28, 0.30, 0.22]] |

## Métricas (columnas de `simulaciones.csv`)

| Columna | Significado |
|---|---|
| `exactitud` | % de días en que Viterbi acierta el estado real |
| `log_verosimilitud` | Log-probabilidad de los síntomas bajo el modelo (Forward) |
| `error_A`, `error_B` | Error absoluto medio entre las matrices estimadas por Baum-Welch y las reales |

## Reproducibilidad

Con las mismas semillas, ejecutar el script dos veces produce archivos CSV idénticos.
Entorno probado: Python 3.13, numpy 2.5, pandas 3.0, matplotlib 3.11.
