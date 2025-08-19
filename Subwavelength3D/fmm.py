from scipy.sparse.linalg import LinearOperator, cg, gmres, minres, bicg, bicgstab, lgmres
import numpy as np
import fmm3dpy as fmm

from scipy.special import spherical_jn, hankel1, sph_harm
from sympy.physics.wigner import wigner_3j

from functools import cache

from joblib import Parallel, delayed, parallel_config


def spherical_hl(n, z):
    return np.sqrt(np.pi / z / 2) * hankel1(n + 1 / 2, z)


def _insulate(f):
    def f_insulated(x):
        x_s = np.squeeze(x)
        N = x_s.shape[0]
        return np.reshape(f(x_s), (N, 1))
    return f_insulated


def get_FMM_operator(centers, radii, eps=1e-3, dipole=False, return_diag=False) -> LinearOperator:
    N = centers.shape[0]

    if not dipole:
        self_interactions = -radii
        source_corrections = -4*np.pi*np.power(radii, 2)

        @_insulate
        def matvec(x: np.ndarray):
            charges = source_corrections * x
            out = fmm.lfmm3d(eps=eps, sources=centers.T,
                             charges=charges, targets=centers.T, pgt=1)
            return out.pottarg + self_interactions*x
        if return_diag:
            return LinearOperator((N, N), matvec=matvec, dtype=float), self_interactions
        return LinearOperator((N, N), matvec=matvec, dtype=float)
    else:
        self_interactions = np.ravel([[-r, -r/3, -r/3, -r/3] for r in radii])
        monopole_source_corrections = -4*np.pi*np.power(radii, 2)
        dipole_source_corrections = -4*np.pi*np.power(radii, 3)/np.sqrt(3)
        dipole_target_corrections = radii/np.sqrt(3)

        @_insulate
        def matvec(x: np.ndarray):
            # x is a vector of shape (4*N,), namely (...,q_i,v_i^1,v_i^2,v_i^3, ...) with monopol coefficient q_i and dipole coefficients v_i^s
            monopoles = monopole_source_corrections * x[0::4]

            v1 = dipole_source_corrections * x[1::4]
            v2 = dipole_source_corrections * x[2::4]
            v3 = dipole_source_corrections * x[3::4]
            dipoles = np.array([v1, v2, v3])

            out = fmm.lfmm3d(eps=eps, sources=centers.T,
                             charges=monopoles, dipvec=dipoles, targets=centers.T, pgt=2)
            monopoles_out = out.pottarg
            dipoles_out = out.gradtarg

            out = np.zeros((4*N,), dtype=float)
            out[0::4] = monopoles_out
            # Same swapping as above
            out[1::4] = dipole_target_corrections * dipoles_out[0]
            out[2::4] = dipole_target_corrections * dipoles_out[1]
            out[3::4] = dipole_target_corrections * dipoles_out[2]
            return out + self_interactions*x

        if return_diag:
            return LinearOperator((4*N, 4*N), matvec=matvec, dtype=float), self_interactions
        return LinearOperator((4*N, 4*N), matvec=matvec, dtype=float)

    # def matmul(X: np.ndarray):
    #     # X is a matrix of shape (N, K) with each column corresponding to a different source arrangement
    #     K = X.shape[1]
    #     out = fmm.hfmm3d(eps=eps, zk=1e-9, sources=centers.T,
    #                      charges=X.T, targets=centers.T, pgt=1, nd=K)
    #     Y = out.pottarg.T  # (N, K)
    #     return -4*np.pi*out.pottarg + self_interactions*X


def operator_to_matrix(F):
    M = F.shape[0]
    S = np.zeros((M, M), dtype=float)
    for j in range(M):
        x = np.zeros(M, dtype=float)
        x[j] = 1.0
        y = F(x)
        S[:, j] = y
    return S


def compute_capacitance_matrix(centers, radii, eps=1e-3, dipole=False):
    N = centers.shape[0]
    S = get_FMM_operator(centers, radii, eps=eps, dipole=dipole)

    def solve_rhs(rhs):
        return gmres(S, rhs, x0=-rhs, tol=1e-5)[0]

    if not dipole:
        C = np.zeros((N, N), dtype=float)
        for j in range(N):
            rhs = np.zeros(N, dtype=float)
            rhs[j] = 1.0
            psi_j = solve_rhs(rhs)
            for i in range(N):
                C[i, j] = - 4*np.pi * radii[i]**2 * psi_j[i]
    else:
        C = np.zeros((N, N), dtype=float)
        for j in range(N):
            rhs = np.zeros(4*N, dtype=float)
            rhs[4*j] = 1.0
            psi_j = solve_rhs(rhs)
            for i in range(N):
                C[i, j] = - 4*np.pi * radii[i]**2 * psi_j[4*i]

    return C


def compute_capacitance_matrix_accelerated(centers, radii, eps=1e-3, dipole=False, n_jobs=12, verbose=False):
    """Fast capacitance matrix calculation using SciPy **CG**, Jacobi PC, and **joblib** parallel RHS.

    - Forms SPD system 	Tilde{S} = S*C (C = diag(1/r^2), repeated for dipoles)
    - Preconditioner: M^{-1} = diag(Tilde{S})^{-1} (Jacobi)
    - Solves each RHS with SciPy's `cg` **in parallel** (joblib)

    Returns
    -------
    C : (N,N) ndarray (float)
    """
    N = centers.shape[0]
    S, S_diag = get_FMM_operator(
        centers, radii, eps=eps, dipole=dipole, return_diag=True)
    M = N if not dipole else 4*N

    # Make single layer potential operator symmetric positive definite by scaling the rows by 1/Rj and taking the negative
    symmetrisation_correction = np.power(radii, -2)
    if dipole:
        symmetrisation_correction = np.repeat(symmetrisation_correction, 4)
    S_tilde = LinearOperator((M, M), matvec=lambda x: -S(
        symmetrisation_correction*x), dtype=float)
    S_tilde_diag = -symmetrisation_correction * S_diag

    Minv = LinearOperator((M, M), matvec=lambda x: x /
                          S_tilde_diag, dtype=float)

    def compute_psi_j(j):
        if verbose:
            if j % 100 == 0:
                print(f"Computing column {j}")
        chi_j = np.zeros(M)
        if not dipole:
            chi_j[j] = 1.0
        else:
            chi_j[4*j] = 1.0
        x, info = cg(S_tilde, chi_j, x0=chi_j, M=Minv, tol=1e-8, maxiter=None)
        if info != 0:
            raise RuntimeError(
                f"CG did not converge for column {j}: info={info}")
        if not dipole:
            psi_j = x
        else:
            psi_j = x[::4]
        return -psi_j

    with parallel_config(backend="loky"):
        cols = Parallel(n_jobs=n_jobs, prefer='processes')(
            delayed(compute_psi_j)(j) for j in range(N)
        )

    Psi = np.column_stack(cols)  # (N, N)
    C = -4*np.pi * Psi
    return C

# Krypy accelerated
# import numpy

# def patch_asscalar(a):
#     return a.item()

# setattr(numpy, "asscalar", patch_asscalar)
# def compute_capacitance_matrix_more_accelerated(centers, radii, eps=1e-3, dipole=False, n_jobs=8):
#     """Fast capacitance matrix calculation using SciPy **CG**, Jacobi PC, and **joblib** parallel RHS.

#     - Forms SPD system 	Tilde{S} = S*C (C = diag(1/r^2), repeated for dipoles)
#     - Preconditioner: M^{-1} = diag(Tilde{S})^{-1} (Jacobi)
#     - Solves each RHS with SciPy's `cg` **in parallel** (joblib)

#     Returns
#     -------
#     C : (N,N) ndarray (float)
#     """
#     N = centers.shape[0]
#     S, S_diag = get_FMM_operator(
#         centers, radii, eps=eps, dipole=dipole, return_diag=True)
#     M = N if not dipole else 4*N

#     # Make single layer potential operator symmetric positive definite by scaling the rows by 1/Rj and taking the negative
#     symmetrisation_correction = np.power(radii, -2)
#     if dipole:
#         symmetrisation_correction = np.repeat(symmetrisation_correction, 4)
#     S_tilde = LinearOperator((M, M), matvec=_insulate(lambda x: -S(
#         symmetrisation_correction*x)), dtype=float)
#     S_tilde_diag = -symmetrisation_correction * S_diag

#     M_op = LinearOperator((M, M), matvec=_insulate(lambda x: x *
#                                                    S_tilde_diag), dtype=float)
#     Minv_op = LinearOperator((M, M), matvec=_insulate(lambda x: x /
#                                                       S_tilde_diag), dtype=float)

#     # Build chunk lists of column indices
#     parts = np.array_split(np.arange(N, dtype=int), max(1, int(n_jobs)))

#     def solve_chunk(idxs: np.ndarray):
#         """Solve a subset of columns sequentially with **recycling**.
#         Returns a list of solution column vectors (each shape (N,))."""
#         if idxs.size == 0:
#             return []
#         # one Recycling CG per chunk; pick an automatic Ritz-based factory
#         rcg = RecyclingCg(vector_factory='RitzApproxKrylov')
#         cols = []
#         for j in idxs:
#             chi_j = np.zeros((M,), dtype=float)
#             if not dipole:
#                 chi_j[j] = 1.0
#             else:
#                 chi_j[4*j] = 1.0
#             # Assemble LinearSystem (self-adjoint + PD per your scaling)
#             lsys = LinearSystem(S_tilde, chi_j, Minv=Minv_op, M=M_op,
#                                 self_adjoint=True, positive_definite=True)
#             # Recycled solve; default tol=1e-8, maxiter=None (N)
#             print(j)
#             sol = rcg.solve(lsys, tol=1e-8, maxiter=None)
#             x = np.squeeze(sol.xk)  # (N,)
#             if not dipole:
#                 psi_j = x
#             else:
#                 psi_j = x[::4]
#             cols.append(psi_j)
#         return cols

#     # with parallel_config(backend='loky'):
#     #     chunk_cols = Parallel(n_jobs=len(parts), prefer='processes')(
#     #         delayed(solve_chunk)(part) for part in parts
#     #     )
#     chunk_cols = []
#     print(f"N: {N}")
#     for part in parts:
#         print(f"Solving chunk {part}")
#         chunk_cols.append(solve_chunk(part))

#     # Reassemble in the correct order
#     Psi = np.zeros((N, N), dtype=float)
#     for part_idxs, part_cols in zip(parts, chunk_cols):
#         for k, j in enumerate(part_idxs):
#             Psi[:, int(j)] = part_cols[k]

#     # Capacitance from monopole potentials
#     C = -4*np.pi * Psi
#     return C
#
