"""
Echo State Network (ESN) - Reservoir Computing

Input -> fixed random reservoir -> trained linear (ridge) readout.
"""

import numpy as np
from sklearn.linear_model import Ridge


class EchoStateNetwork:
    """
    Echo State Network (ESN).

    Only the readout layer is trained; the reservoir weights stay fixed.
    The reservoir maps the input into a high-dimensional state space from
    which a linear readout extracts predictions.

    Hardware motivation:
        Fixed reservoir weights can be encoded in non-volatile memristive
        crossbar arrays for in-memory, low-power inference at the edge.

    Parameters
    ----------
    input_dim : int
    reservoir_dim : int
    spectral_radius : float
        Target largest |eigenvalue| of the reservoir matrix. Values below 1
        are the usual heuristic for the echo state property (it is not a
        strict guarantee when the reservoir is driven by input).
    sparsity : float
        Fraction of reservoir weights set to zero.
    noise_level : float
        Std of noise added to the state update (0 disables it).
    leak_rate : float in (0, 1]
        1.0 gives the standard ESN; smaller values give a leaky integrator.
    ridge_alpha : float
        Ridge regularization of the readout.
    random_seed : int
        Seeds weights AND state noise, so runs are reproducible.
    """

    def __init__(self, input_dim, reservoir_dim=500, spectral_radius=0.9,
                 sparsity=0.9, noise_level=0.001, leak_rate=1.0,
                 ridge_alpha=1.0, random_seed=42):
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise_level = noise_level
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha

        self.rng = np.random.RandomState(random_seed)
        self._init_weights()

        self.state = np.zeros(reservoir_dim)
        self.readout = Ridge(alpha=ridge_alpha)
        self._is_trained = False
        self.classes_ = None  # set when trained on integer class labels

    # ------------------------------------------------------------------
    def _init_weights(self):
        """W_in ~ U(-1, 1); W_res sparse random, rescaled to spectral_radius."""
        self.W_in = self.rng.uniform(-1, 1, (self.reservoir_dim, self.input_dim))

        W = self.rng.uniform(-1, 1, (self.reservoir_dim, self.reservoir_dim))
        W[self.rng.rand(*W.shape) < self.sparsity] = 0.0

        max_eig = np.max(np.abs(np.linalg.eigvals(W)))
        if max_eig == 0:
            raise ValueError("Reservoir matrix is all zeros - reduce sparsity.")
        self.W_res = W * (self.spectral_radius / max_eig)

    # ------------------------------------------------------------------
    def reset_state(self):
        self.state = np.zeros(self.reservoir_dim)

    def _update_state(self, x):
        """h(t) = (1-a) h(t-1) + a * tanh(W_in x + W_res h(t-1) + noise)"""
        pre = self.W_in @ x + self.W_res @ self.state
        if self.noise_level:
            pre = pre + self.noise_level * self.rng.randn(self.reservoir_dim)
        a = self.leak_rate
        self.state = (1 - a) * self.state + a * np.tanh(pre)
        return self.state.copy()

    def _collect_states(self, X, washout=0, reset=True):
        """Drive the reservoir with X; drop the first `washout` states."""
        if reset:
            self.reset_state()
        states = [self._update_state(x) for x in X]
        return np.array(states)[washout:]

    # ------------------------------------------------------------------
    @staticmethod
    def _as_2d(y):
        y = np.asarray(y)
        return y.reshape(len(y), -1)

    def fit(self, X, y, washout=50, verbose=True):
        """
        Train the readout.

        y may be real-valued (regression) or a 1-D array of integer class
        labels (classification, one-hot encoded internally).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim == 1 and np.issubdtype(y.dtype, np.integer):
            self.classes_ = np.unique(y)
            Y = (y[:, None] == self.classes_[None, :]).astype(float)
        else:
            self.classes_ = None
            Y = self._as_2d(y)

        states = self._collect_states(X, washout=washout)
        self.readout.fit(states, Y[washout:])
        self._is_trained = True
        if verbose:
            print(f"[ESN] Trained: {self.reservoir_dim} neurons, "
                  f"Ridge(alpha={self.ridge_alpha})")
        return self

    def predict(self, X, washout=0, reset=True):
        """Always returns an array of shape (T - washout, n_outputs)."""
        if not self._is_trained:
            raise RuntimeError("ESN is not trained yet. Call .fit() first.")
        states = self._collect_states(np.asarray(X), washout=washout, reset=reset)
        return self._as_2d(self.readout.predict(states))

    def predict_classes(self, X, washout=0):
        return self.classes_[np.argmax(self.predict(X, washout=washout), axis=1)]

    # ------------------------------------------------------------------
    def score(self, X, y, task="regression", washout=0, verbose=True):
        """task='regression' -> NMSE (lower is better); 'classification' -> accuracy."""
        y = np.asarray(y)
        if task == "classification":
            preds = self.predict_classes(X, washout=washout)
            acc = float(np.mean(preds == y[washout:]))
            if verbose:
                print(f"[ESN] Accuracy: {acc * 100:.2f}%")
            return acc
        if task == "regression":
            preds = self.predict(X, washout=washout)
            y_eval = self._as_2d(y)[washout:]          # same shape as preds
            nmse = float(np.mean((preds - y_eval) ** 2) / np.var(y_eval))
            if verbose:
                print(f"[ESN] NMSE: {nmse:.6g}")
            return nmse
        raise ValueError("task must be 'classification' or 'regression'.")

    def summary(self):
        print("=" * 50)
        print("       Echo State Network - Summary")
        print("=" * 50)
        print(f"  Input dimension    : {self.input_dim}")
        print(f"  Reservoir neurons  : {self.reservoir_dim}")
        print(f"  Spectral radius    : {self.spectral_radius}")
        print(f"  Sparsity           : {self.sparsity * 100:.0f}%")
        print(f"  Noise level        : {self.noise_level}")
        print(f"  Leak rate          : {self.leak_rate}")
        print(f"  Readout            : Ridge (alpha={self.ridge_alpha})")
        print(f"  Trained            : {self._is_trained}")
        print("=" * 50)


# ======================================================================
# Demo: one-step-ahead sine prediction
# ======================================================================
if __name__ == "__main__":
    T = 1000
    t = np.linspace(0, 10 * np.pi, T)
    signal = np.sin(t).reshape(-1, 1)
    X, y = signal[:-1], signal[1:]

    split = 700
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    esn = EchoStateNetwork(input_dim=1, reservoir_dim=300, spectral_radius=0.9,
                           sparsity=0.9, noise_level=0.001, ridge_alpha=1e-4)
    esn.summary()
    esn.fit(X_train, y_train, washout=50)
    esn.score(X_test, y_test, task="regression")
