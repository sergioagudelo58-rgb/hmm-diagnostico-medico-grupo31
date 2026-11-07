"""
Modelo Oculto de Markov (HMM) discreto para diagnóstico médico.

Estados ocultos: Sano, Gripe, Neumonía (lo que NO vemos).
Observaciones:   Sin síntomas, Tos, Fiebre, Fatiga (lo que SÍ vemos).

El modelo se define con tres piezas:
  pi -> probabilidad de cómo empieza el paciente
  A  -> probabilidad de pasar de un estado a otro de un día al siguiente
  B  -> probabilidad de ver cada síntoma estando en cada estado
"""
import itertools
import numpy as np

ESTADOS = ["Sano", "Gripe", "Neumonía"]
SINTOMAS = ["Sin síntomas", "Tos", "Fiebre", "Fatiga"]


class HMM:
    def __init__(self, pi, A, B):
        self.pi = np.array(pi, dtype=float)
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.N = self.A.shape[0]   # número de estados
        self.M = self.B.shape[1]   # número de síntomas
        self.validar()

    # ------------------------------------------------------------------
    def validar(self):
        """Revisa que pi y cada fila de A y B sumen 1 (son probabilidades)."""
        if not np.isclose(self.pi.sum(), 1):
            raise ValueError("pi no suma 1")
        if not np.allclose(self.A.sum(axis=1), 1):
            raise ValueError("Alguna fila de A no suma 1")
        if not np.allclose(self.B.sum(axis=1), 1):
            raise ValueError("Alguna fila de B no suma 1")

    # ------------------------------------------------------------------
    def simular(self, T, semilla):
        """Genera T días: el estado real de cada día y el síntoma observado."""
        rng = np.random.default_rng(semilla)
        estados = np.zeros(T, dtype=int)
        obs = np.zeros(T, dtype=int)
        estados[0] = rng.choice(self.N, p=self.pi)          # día 1 según pi
        obs[0] = rng.choice(self.M, p=self.B[estados[0]])   # síntoma del día 1
        for t in range(1, T):
            estados[t] = rng.choice(self.N, p=self.A[estados[t - 1]])  # cambio de estado
            obs[t] = rng.choice(self.M, p=self.B[estados[t]])          # síntoma del día
        return estados, obs

    # ------------------------------------------------------------------
    def forward(self, obs):
        """Algoritmo Forward: qué tan probable es ver esos síntomas.
        Usa escalado para no trabajar con números demasiado pequeños.
        Devuelve la log-verosimilitud, las alfas escaladas y los factores de escala."""
        T = len(obs)
        alfa = np.zeros((T, self.N))
        c = np.zeros(T)
        alfa[0] = self.pi * self.B[:, obs[0]]
        c[0] = alfa[0].sum()
        alfa[0] /= c[0]
        for t in range(1, T):
            alfa[t] = (alfa[t - 1] @ self.A) * self.B[:, obs[t]]
            c[t] = alfa[t].sum()
            alfa[t] /= c[t]
        return np.log(c).sum(), alfa, c

    def _backward(self, obs, c):
        """Paso hacia atrás (lo necesita Baum-Welch), con la misma escala de Forward."""
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
        return beta

    # ------------------------------------------------------------------
    def viterbi(self, obs):
        """Algoritmo de Viterbi: la secuencia de estados más probable
        dados los síntomas. Trabaja con logaritmos para evitar números diminutos."""
        T = len(obs)
        logA, logB, logpi = np.log(self.A), np.log(self.B), np.log(self.pi)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = logpi + logB[:, obs[0]]
        for t in range(1, T):
            candidatos = delta[t - 1][:, None] + logA      # todas las formas de llegar
            psi[t] = candidatos.argmax(axis=0)              # mejor estado anterior
            delta[t] = candidatos.max(axis=0) + logB[:, obs[t]]
        # recorrer el camino de atrás hacia adelante
        camino = np.zeros(T, dtype=int)
        camino[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            camino[t] = psi[t + 1][camino[t + 1]]
        return camino


# ----------------------------------------------------------------------
def baum_welch(obs, N, M, semilla, max_iter=100, tol=1e-4):
    """Algoritmo Baum-Welch: intenta APRENDER pi, A y B solo a partir de los
    síntomas, empezando desde valores al azar (con semilla fija)."""
    rng = np.random.default_rng(semilla)
    modelo = HMM(rng.dirichlet(np.ones(N)),
                 rng.dirichlet(np.ones(N), size=N),
                 rng.dirichlet(np.ones(M), size=N))
    T = len(obs)
    log_v_anterior = -np.inf
    for _ in range(max_iter):
        log_v, alfa, c = modelo.forward(obs)
        beta = modelo._backward(obs, c)
        gamma = alfa * beta                          # prob. de cada estado cada día
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((N, N))                        # prob. de cada cambio de estado
        for t in range(T - 1):
            x = alfa[t][:, None] * modelo.A * (modelo.B[:, obs[t + 1]] * beta[t + 1])[None, :]
            xi += x / x.sum()
        # actualizar parámetros
        pi = gamma[0]
        A = xi / gamma[:-1].sum(axis=0)[:, None]
        B = np.zeros((N, M))
        for k in range(M):
            B[:, k] = gamma[obs == k].sum(axis=0)
        B /= gamma.sum(axis=0)[:, None]
        B = np.clip(B, 1e-10, None)
        B /= B.sum(axis=1, keepdims=True)
        modelo = HMM(pi / pi.sum(), A / A.sum(axis=1, keepdims=True), B)
        if abs(log_v - log_v_anterior) < tol:
            break
        log_v_anterior = log_v
    return modelo


def alinear_estados(estimado, real):
    """Baum-Welch no sabe qué estado se llama 'Sano' o 'Gripe': puede devolverlos
    en otro orden. Probamos todos los órdenes y nos quedamos con el que más se
    parece al modelo real, para poder comparar."""
    mejor, mejor_error = None, np.inf
    for p in itertools.permutations(range(real.N)):
        p = list(p)
        A_p = estimado.A[np.ix_(p, p)]
        B_p = estimado.B[p]
        error = np.abs(A_p - real.A).mean() + np.abs(B_p - real.B).mean()
        if error < mejor_error:
            mejor, mejor_error = (A_p, B_p), error
    return mejor
