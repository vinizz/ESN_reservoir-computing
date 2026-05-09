"""
Echo State Network (ESN) — Reservoir Computing

"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score


class EchoStateNetwork:
    """
    Echo State Network (ESN) — Reservoir Computing

    Architecture: Input → Fixed Random Reservoir → Trained Readout

    Key Principle:
        Only the readout (output) layer is trained.
        The reservoir weights remain fixed after initialization.
        The reservoir projects inputs into a rich, high-dimensional
        feature space from which a linear readout extracts predictions.

    Hardware Motivation:
        ESNs naturally map to memristive crossbar arrays for in-memory
        computing — fixed reservoir weights can be physically encoded
        into non-volatile memristive devices, enabling ultra-low-power
        neuromorphic inference at the edge.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input signal.
    reservoir_dim : int
        Number of neurons in the reservoir (default: 500).
        Larger reservoirs capture richer dynamics but cost more compute.
    spectral_radius : float
        Largest eigenvalue of the reservoir weight matrix (default: 0.9).
        Controls the "echo" — how long past inputs influence the state.
        Must be < 1.0 for the Echo State Property (ESP) to hold.
    sparsity : float
        Fraction of reservoir connections set to zero (default: 0.9).
        Sparse reservoirs better mimic biological neural networks and
        are more efficient to implement in hardware crossbar arrays.
    noise_level : float
        Small noise added during state update for regularization (default: 0.001).
    output_dim : int
        Number of output classes or regression targets (default: 1).
    ridge_alpha : float
        Regularization strength for the Ridge Regression readout (default: 1.0).
        Higher values prevent overfitting on small datasets.
    """

    def __init__(
        self,
        input_dim,
        reservoir_dim=500,
        spectral_radius=0.9,
        sparsity=0.9,
        noise_level=0.001,
        output_dim=1,
        ridge_alpha=1.0,
        random_seed=42,
    ):
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise_level = noise_level
        self.output_dim = output_dim
        self.ridge_alpha = ridge_alpha

        rng = np.random.RandomState(random_seed)
        self._init_weights(rng)

        # Reservoir state — updated at every time step
        self.state = np.zeros(reservoir_dim)

        # Readout layer — trained via Ridge Regression
        self.readout = Ridge(alpha=ridge_alpha)
        self._is_trained = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self, rng):
        """
        Initialize input and reservoir weight matrices.

        Input weights (W_in):
            Randomly drawn from Uniform(-1, 1).
            Scales how strongly each input drives reservoir neurons.

        Reservoir weights (W_res):
            Sparse random matrix, scaled so its spectral radius
            matches the target value.
            Spectral radius < 1  →  Echo State Property guaranteed.
        """
        # Input weight matrix: shape (reservoir_dim, input_dim)
        self.W_in = rng.uniform(-1, 1, (self.reservoir_dim, self.input_dim))

        # Reservoir weight matrix: shape (reservoir_dim, reservoir_dim)
        W = rng.uniform(-1, 1, (self.reservoir_dim, self.reservoir_dim))

        # Apply sparsity mask
        mask = rng.rand(self.reservoir_dim, self.reservoir_dim) < self.sparsity
        W[mask] = 0.0

        # Rescale to target spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig == 0:
            raise ValueError("Reservoir matrix is all zeros — reduce sparsity.")
        self.W_res = W * (self.spectral_radius / max_eig)

    # ------------------------------------------------------------------
    # Core Reservoir Dynamics
    # ------------------------------------------------------------------

    def reset_state(self):
        """Reset reservoir state to zero (call before each new sequence)."""
        self.state = np.zeros(self.reservoir_dim)

    def _update_state(self, x):
        """
        One-step reservoir state update.

        Equation:
            h(t) = tanh( W_in · x(t)  +  W_res · h(t-1)  +  noise )

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Current input vector.

        Returns
        -------
        np.ndarray, shape (reservoir_dim,)
            Updated reservoir state.
        """
        input_drive    = self.W_in  @ x           # Input contribution
        recurrent_drive = self.W_res @ self.state  # Recurrent (memory) contribution
        noise          = self.noise_level * np.random.randn(self.reservoir_dim)

        self.state = np.tanh(input_drive + recurrent_drive + noise)
        return self.state.copy()

    def _collect_states(self, X, washout=50):
        """
        Drive the reservoir with input X and collect reservoir states.

        The washout period discards the first `washout` steps so the
        reservoir settles from its arbitrary initial state.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_dim)  — time-series input
            OR shape (N, input_dim)            — batch of static vectors
        washout : int
            Number of initial time steps to discard (for time-series).

        Returns
        -------
        np.ndarray, shape (T - washout, reservoir_dim)
            Collected reservoir states.
        """
        self.reset_state()
        states = []
        for t, x in enumerate(X):
            state = self._update_state(x)
            if t >= washout:
                states.append(state)
        return np.array(states)

    # ------------------------------------------------------------------
    # Training & Inference
    # ------------------------------------------------------------------

    def fit(self, X, y, washout=50):
        """
        Train the ESN readout layer.

        Only Ridge Regression on the collected reservoir states is trained.
        Reservoir weights W_in and W_res are NOT modified.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_dim)
            Input time series or batch of input vectors.
        y : np.ndarray, shape (T,) or (T, output_dim)
            Target labels or values (washout samples will be trimmed).
        washout : int
            Steps to discard at the start (for time-series inputs).

        Returns
        -------
        self
        """
        states = self._collect_states(X, washout=washout)

        # Trim y to match states after washout
        y_trimmed = y[washout:] if len(y) > len(states) else y

        self.readout.fit(states, y_trimmed)
        self._is_trained = True
        print(f"[ESN] Training complete — reservoir: {self.reservoir_dim} neurons, "
              f"readout: Ridge(alpha={self.ridge_alpha})")
        return self

    def predict(self, X, washout=0):
        """
        Generate predictions from trained ESN.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_dim)
        washout : int
            Washout steps for test sequences (usually 0 for static inputs).

        Returns
        -------
        np.ndarray — predicted values or class indices.
        """
        if not self._is_trained:
            raise RuntimeError("ESN is not trained yet. Call .fit() first.")

        states = self._collect_states(X, washout=washout)
        return self.readout.predict(states)

    def predict_classes(self, X, washout=0):
        """
        Predict class labels (argmax over readout outputs).
        Use for classification tasks.
        """
        raw = self.predict(X, washout=washout)
        return np.argmax(raw, axis=1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def score(self, X, y, task="classification", washout=0):
        """
        Evaluate ESN performance.

        Parameters
        ----------
        task : str
            'classification' → accuracy
            'regression'     → NMSE (Normalized Mean Square Error)

        Returns
        -------
        float — accuracy or NMSE.
        """
        if task == "classification":
            preds = self.predict_classes(X, washout=washout)
            y_eval = y[washout:] if len(y) > len(preds) else y
            acc = accuracy_score(y_eval, preds)
            print(f"[ESN] Classification Accuracy: {acc * 100:.2f}%")
            return acc

        elif task == "regression":
            preds = self.predict(X, washout=washout)
            y_eval = y[washout:] if len(y) > len(preds) else y
            nmse = np.mean((preds - y_eval) ** 2) / np.var(y_eval)
            print(f"[ESN] NMSE: {nmse:.6f}")
            return nmse

        else:
            raise ValueError("task must be 'classification' or 'regression'.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self):
        """Print ESN architecture summary."""
        print("=" * 50)
        print("       Echo State Network — Summary")
        print("=" * 50)
        print(f"  Input dimension    : {self.input_dim}")
        print(f"  Reservoir neurons  : {self.reservoir_dim}")
        print(f"  Spectral radius    : {self.spectral_radius}")
        print(f"  Sparsity           : {self.sparsity * 100:.0f}%")
        print(f"  Noise level        : {self.noise_level}")
        print(f"  Readout            : Ridge Regression (alpha={self.ridge_alpha})")
        print(f"  Trained            : {self._is_trained}")
        print("=" * 50)
        print("  Hardware mapping   : Reservoir weights encodable")
        print("  onto memristive crossbar arrays for in-memory compute.")
        print("=" * 50)


# ======================================================================
# Quick Demo — Sine Wave Prediction (Regression)
# ======================================================================

if __name__ == "__main__":

    print("\n--- ESN Demo: Sine Wave Prediction ---\n")

    # Generate a sine wave time series
    T = 1000
    t = np.linspace(0, 10 * np.pi, T)
    signal = np.sin(t).reshape(-1, 1)

    # Input: current value, Target: next value (one-step-ahead prediction)
    X = signal[:-1]   # shape (999, 1)
    y = signal[1:]    # shape (999, 1)

    # Split train / test
    split = 700
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # Build and train ESN
    esn = EchoStateNetwork(
        input_dim=1,
        reservoir_dim=300,
        spectral_radius=0.9,
        sparsity=0.9,
        noise_level=0.001,
        ridge_alpha=1e-4,
    )

    esn.summary()
    esn.fit(X_train, y_train, washout=50)

    # Evaluate
    esn.score(X_test, y_test, task="regression")

    # Predict
    preds = esn.predict(X_test)
    nmse = np.mean((preds - y_test) ** 2) / np.var(y_test)
    print(f"\nPrediction NMSE on test set : {nmse:.6f}")
    print("(Lower is better — NMSE < 0.01 is excellent for ESNs)\n")
