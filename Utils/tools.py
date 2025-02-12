import numpy as np


def tfbt(v: np.array, k: int) -> np.array:
    """
    Computes the truncated Floquet-Bloch transform of the column vectors of v assuming a unit-cell with k particles.

    Args:
        v (np.array): one or 2 dimensional array, tfbt is computed on the columns (aka axis 1)
        k (int): number of particles is a unit cell, v.shape[0] must be a multiple of k

    Returns:
        np.array: Truncated Floquet-Bloch transform of the column of v.
                  The shape is (v.shape[0] // k, v.shape[1], k)

    The TFBT is defined in Definition 4.2 of https://arxiv.org/pdf/2410.17597.
    """
    if v.shape[0] % k != 0:
        ValueError(f"v.shape[0](= {v.shape[0]}) must be a multiple of k(={k})")

    grouped = np.zeros((v.shape[0] // k, v.shape[1], k))
    # Regroup entries according to their position in the unit cell
    for i in range(k):
        idx = np.arange(i, v.shape[0], k)
        grouped[:, :, i] = v[idx, :]

    return np.fft.fft(grouped, axis=0) / np.sqrt(v.shape[0] // k)


def recover_quasiperiodicity(tfbt: np.array) -> np.array:
    """
    Computes the discrete quasiperiodicity from the truncated Floquet-Bloch transform

    Args:
        tfbt (np.array): truncated Floquet-Bloch transform, output of tfbt

    Returns:
        np.array: quasiperiodicity for every vector in the output of tfbt.
        The shape is (tfbt(v,k).shape[1]) = (v.shape[1])

    The Discrete quasiperiodicity is defined in (Definition 4.4)
    """
    N = tfbt.shape[0]
    sliced = np.abs(tfbt)
    proj = np.sum(np.power(sliced, 2), axis=-1)

    # Precompute indices to match the other of the fft
    indices = np.abs(np.arange(N) - (N // 2)) / N
    # Averaging over the indices and scaling
    mult = (
        np.multiply(proj[np.arange(N) - (N // 2)], indices[:, np.newaxis]) * 2 * np.pi
    )
    return np.sum(mult, axis=0)
