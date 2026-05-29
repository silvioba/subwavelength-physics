"""Floquet-Bloch transform, generalised Brillouin zone, and Laurent polynomial utilities."""

import numpy as np
from typing import Tuple


def tfbt(v: np.array, k: int) -> np.array:
    """
    Computes the truncated Floquet-Bloch transform of the column vectors of v assuming a unit-cell with k particles.

    Args:
        v (np.array): one or 2 dimensional array, tfbt is computed on the columns (aka axis 1)
        k (int): number of particles is a unit cell, v.shape[0] must be a multiple of k

    Returns:
        np.array: Truncated Floquet-Bloch transform of the column of v.
                  The shape is (v.shape[0] // k, v.shape[1], k)

    The TFBT is defined in Definition 3.2 of the TLMS paper (Ammari, Barandun, Uhlmann 2025).
    """
    if v.shape[0] % k != 0:
        raise ValueError(f"v.shape[0](= {v.shape[0]}) must be a multiple of k(={k})")

    grouped = np.zeros((v.shape[0] // k, v.shape[1], k), dtype=v.dtype)
    # Regroup entries according to their position in the unit cell
    for i in range(k):
        idx = np.arange(i, v.shape[0], k)
        grouped[:, :, i] = v[idx, :]

    return np.fft.fft(grouped, axis=0) / np.sqrt(v.shape[0] // k)


def recover_quasiperiodicity(tfbt: np.array) -> np.array:
    """
    Computes the discrete quasiperiodicity from the truncated Floquet-Bloch transform.

    This returns a weighted average of |alpha_j| (Definition 3.4 of the TLMS paper):
        Q_m(u) = sum_{alpha_j in Y_m^*} |alpha_j| * ||T^j(u)||^2 / ||u||^2

    The normalization by ||u||^2 ensures the result is scale-invariant.

    Args:
        tfbt (np.array): truncated Floquet-Bloch transform, output of tfbt

    Returns:
        np.array: quasiperiodicity for every vector in the output of tfbt.
        The shape is (tfbt(v,k).shape[1]) = (v.shape[1])
    """
    N = tfbt.shape[0]
    sliced = np.abs(tfbt)
    proj = np.sum(np.power(sliced, 2), axis=-1)

    # Precompute indices to match the order of the fft
    indices = np.abs(np.arange(N) - (N // 2)) / N
    # Averaging over the indices and scaling
    mult = (
        np.multiply(proj[np.arange(N) - (N // 2)], indices[:, np.newaxis]) * 2 * np.pi
    )
    # Normalize by total power (= ||u||^2 by Parseval) for scale invariance
    total_power = np.sum(proj, axis=0)
    total_power = np.where(total_power > 0, total_power, 1.0)  # avoid division by zero
    return np.sum(mult, axis=0) / total_power


def estimate_exponential_decay(u: np.ndarray, d: float = 1.0, decay_cutoff: int = 10) -> float:
    """Estimate exponential decay rate beta from an eigenmode.

    Fits log|u_n| = -beta * n * d + const via least squares.
    Positive beta means the mode decays in the +x direction.

    Args:
        u: 1D complex array (eigenmode of length N).
        d: Lattice spacing.
        decay_cutoff: Number of entries to discard from each end to reduce
            edge effects. Default 10. Set to 0 to use all entries.

    Returns:
        float: Estimated decay rate beta.
    """
    N = len(u)
    lo = min(decay_cutoff, N // 3)
    hi = max(N - lo, lo + 1)
    u_interior = u[lo:hi]

    amp = np.abs(u_interior)
    mask = amp > 1e-15
    if np.sum(mask) < 3:
        return 0.0

    x = (np.arange(lo, hi)[mask]) * d
    log_amp = np.log(amp[mask])

    slope, _ = np.polyfit(x, log_amp, 1)
    return -slope


def demodulate_eigenmode(u: np.ndarray, beta: float, d: float = 1.0) -> np.ndarray:
    """Remove exponential envelope from an eigenmode.

    Computes v_n = u_n * exp(beta * n * d) to cancel the exp(-beta * n * d) decay.

    Args:
        u: 1D complex array (eigenmode).
        beta: Decay rate (from estimate_exponential_decay).
        d: Lattice spacing.

    Returns:
        np.ndarray: Demodulated mode v (should be approximately periodic).
    """
    n = np.arange(len(u))
    return u * np.exp(beta * n * d)


def recover_quasiperiodicity_peak(tfbt_result: np.ndarray) -> np.ndarray:
    """Extract the dominant quasiperiodicity alpha from TFBT output.

    Unlike recover_quasiperiodicity() which returns a weighted average,
    this returns the alpha at which ||T^j(u)||^2 is maximised.

    Args:
        tfbt_result: Output of tfbt(), shape (N_cells, num_vectors, k).

    Returns:
        np.ndarray: Array of alpha values (one per vector), in [-pi, pi).
    """
    N = tfbt_result.shape[0]
    num_vectors = tfbt_result.shape[1]

    # Power spectrum: sum over particles in unit cell
    power = np.sum(np.abs(tfbt_result)**2, axis=-1)  # (N, num_vectors)

    # FFT frequency ordering: np.fft.fft gives indices 0, 1, ..., N-1
    # corresponding to alpha_j = 2*pi*j/N, but we want [-pi, pi)
    freqs = np.fft.fftfreq(N) * 2 * np.pi  # alpha values in [-pi, pi)

    alphas = np.zeros(num_vectors)
    for j in range(num_vectors):
        peak_idx = np.argmax(power[:, j])
        alphas[j] = freqs[peak_idx]

    return alphas


def get_laurent_polynomial(coeffs: np.ndarray) -> callable:
    """Return a Laurent polynomial as a callable from its Fourier coefficients.

    Given coefficients c_{-k}, ..., c_0, ..., c_k, returns the function

        p(z) = sum_{m=-k}^{k} c_m * z^m

    The coefficients array is indexed so that coeffs[0] corresponds to
    m = -k and coeffs[2k] corresponds to m = k.

    For matrix-valued coefficients (e.g. from
    ClassicPeriodicFWP3D.get_toeplitz_coefficients), each coeffs[j] is
    an (N, N) matrix and the returned function evaluates a matrix-valued
    Laurent polynomial.

    Args:
        coeffs: Array of shape (2k+1, ...) where coeffs[j] is the
            coefficient for m = -k + j.

    Returns:
        Callable that maps a scalar or array z to the Laurent polynomial
        value. For scalar coefficients returns a scalar/array matching z.
        For matrix coefficients returns shape (*z.shape, N, N) or (N, N)
        for scalar z.
    """
    n_coeffs = coeffs.shape[0]
    k = (n_coeffs - 1) // 2
    ms = np.arange(-k, k + 1)

    def p(z):
        z = np.asarray(z)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        # z_powers[j, i] = z_i^{m_j}
        z_powers = z[None, :] ** ms[:, None]  # (2k+1, len(z))

        if coeffs.ndim == 1:
            # Scalar coefficients
            result = coeffs @ z_powers  # (len(z),)
            return result[0] if scalar_input else result
        else:
            # Matrix coefficients of shape (2k+1, N, N)
            # result[i] = sum_j coeffs[j] * z_i^{m_j}
            result = np.einsum('jab,ji->iab', coeffs, z_powers)
            return result[0] if scalar_input else result

    return p


def estimate_GBZ(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    k: int = 1,
    d: float = 1.0,
    method: str = "averaged",
    decay_cutoff: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the generalised Brillouin zone from finite eigenmodes.

    For each eigenmode u_i with eigenvalue lambda_i:
      1. Estimate exponential decay beta_i (skin effect)
      2. Demodulate to remove the envelope
      3. Apply TFBT and extract alpha_i (quasiperiodicity)

    Args:
        eigenvalues: Array of eigenvalues (length N_modes).
        eigenvectors: Matrix of eigenvectors (columns), shape (N, N_modes).
        k: Number of resonators per unit cell.
        d: Lattice spacing.
        method: How to extract alpha from the TFBT spectrum.
            "averaged" (default): weighted average Q_m from Definition 3.4
                (recover_quasiperiodicity). More robust, smooths over noise.
            "peak": argmax of the TFBT power spectrum
                (recover_quasiperiodicity_peak). Sharper but noisier.
        decay_cutoff: Number of entries to discard from each end when
            estimating beta, to reduce edge effects. Default 10.

    Returns:
        Tuple of (alphas, betas, eigenvalues), each of length N_modes.
    """
    N_modes = len(eigenvalues)
    alphas = np.zeros(N_modes)
    betas = np.zeros(N_modes)

    for i in range(N_modes):
        u = eigenvectors[:, i]

        # Step 1: estimate decay
        betas[i] = estimate_exponential_decay(u, d, decay_cutoff=decay_cutoff)

        # Step 2: demodulate
        v = demodulate_eigenmode(u, betas[i], d)

        # Step 3: TFBT and extract alpha
        v_2d = v.reshape(-1, 1)
        tfbt_result = tfbt(v_2d, k)

        if method == "peak":
            alphas[i] = recover_quasiperiodicity_peak(tfbt_result)[0]
        else:
            alphas[i] = recover_quasiperiodicity(tfbt_result)[0]

    return alphas, betas, eigenvalues
