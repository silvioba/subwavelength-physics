import unittest
import numpy as np
from parameterized import parameterized

from Subwavelength3D.continuum import (
    ContinuumFiniteSWP3D,
    _complex_to_real_sh_unitary,
    _spherical_jn_prime_zeros,
    _neumann_trace_kappa,
)
from Subwavelength3D.classic_finite import ClassicFiniteSWP3D


def _chain(N: int, spacing: float = 3.0, R: float = 1.0, v_in: float = 1.0):
    centers = np.array([[0.0, 0.0, i * spacing] for i in range(N)])
    radii = R * np.ones(N)
    v = v_in * np.ones(N)
    return centers, radii, v


class AssemblySanity(unittest.TestCase):

    def test_A_matrix_shape_and_finite(self):
        centers, radii, v_in = _chain(N=3)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        N_multipole = 2
        A = cn.get_A_matrix(omega=0.1 + 0j, delta=1e-3, N_multipole=N_multipole)
        expected_dim = 2 * 3 * N_multipole ** 2
        self.assertEqual(A.shape, (expected_dim, expected_dim))
        self.assertTrue(np.all(np.isfinite(A)))

    def test_delta_broadcast(self):
        """Scalar delta and uniform-array delta should produce identical A."""
        centers, radii, v_in = _chain(N=3)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        A_scalar = cn.get_A_matrix(omega=0.1 + 0j, delta=1e-3, N_multipole=1)
        A_array = cn.get_A_matrix(omega=0.1 + 0j, delta=np.full(3, 1e-3), N_multipole=1)
        np.testing.assert_allclose(A_scalar, A_array)


class SubwavelengthRecovery(unittest.TestCase):
    """The exact solver must recover ClassicFiniteSWP3D in the delta -> 0 limit.

    At leading order, omega_j ~ sqrt(delta * lambda_j(VC)) for the N lowest
    resonances, where lambda_j are eigenvalues of V*C from the classical
    (generalised-capacitance) formulation.
    """

    @parameterized.expand([
        [2, 1e-4, 1],
        [3, 1e-4, 1],
        [3, 1e-5, 1],
    ])
    def test_subwavelength_roots_match_leading_order(self, N, delta, N_multipole):
        centers, radii, v_in = _chain(N=N)

        cl = ClassicFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        lam, _ = cl.compute_sorted_eigs_capacitance_matrix(
            N_multipole=N_multipole, method='general', sorting='eva_real'
        )
        omega_lead = np.sqrt(delta * lam)

        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        x0 = 0.5 * omega_lead[0] + 0j  # start below smallest predicted root
        roots = cn.compute_resonances(
            x0=x0, N_roots=N, delta=delta,
            N_multipole=N_multipole, perturbation=1e-4 * np.sqrt(delta),
        )

        roots_re_sorted = np.sort(roots.real)
        # Real parts must agree with the leading-order prediction to <1% relative.
        np.testing.assert_allclose(
            roots_re_sorted, omega_lead, rtol=1e-2,
            err_msg=f"delta={delta}, N={N}: Re(roots)={roots_re_sorted}, expected={omega_lead}",
        )
        # Imaginary parts encode the first-order radiative correction; they are
        # small but non-zero. Require |Im| / |Re| < 10% for safety.
        np.testing.assert_array_less(
            np.abs(roots.imag) / np.abs(roots.real), 0.1,
            err_msg=f"Damping ratio too large: {roots}",
        )

    def test_delta_scaling(self):
        """Halving delta should shrink the lowest omega by ~ sqrt(2)."""
        centers, radii, v_in = _chain(N=2)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        cl = ClassicFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        lam, _ = cl.compute_sorted_eigs_capacitance_matrix(
            N_multipole=1, method='general', sorting='eva_real'
        )

        def lowest_root(delta):
            x0 = 0.5 * np.sqrt(delta * lam[0]) + 0j
            roots = cn.compute_resonances(
                x0=x0, N_roots=1, delta=delta,
                N_multipole=1, perturbation=1e-4 * np.sqrt(delta),
            )
            return roots[0].real

        r1 = lowest_root(1e-4)
        r2 = lowest_root(5e-5)
        self.assertAlmostEqual(r1 / r2, np.sqrt(2.0), delta=0.05)


class MultipoleConvergence(unittest.TestCase):

    def test_N_multipole_2_vs_3(self):
        """Higher-order multipole refinements should stabilise.

        Monopole->dipole carries a genuine O(sqrt(delta)) physical correction,
        but dipole->quadrupole should be much smaller (subwavelength: each
        extra l contributes an additional (kR)^2 factor).
        """
        centers, radii, v_in = _chain(N=2)
        delta = 1e-4
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        cl = ClassicFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        lam, _ = cl.compute_sorted_eigs_capacitance_matrix(
            N_multipole=1, method='general', sorting='eva_real'
        )
        x0 = 0.5 * np.sqrt(delta * lam[0]) + 0j

        roots_L2 = cn.compute_resonances(
            x0=x0, N_roots=2, delta=delta, N_multipole=2,
            perturbation=1e-4 * np.sqrt(delta),
        )
        roots_L3 = cn.compute_resonances(
            x0=x0, N_roots=2, delta=delta, N_multipole=3,
            perturbation=1e-4 * np.sqrt(delta),
        )
        r2 = np.sort(roots_L2.real)
        r3 = np.sort(roots_L3.real)
        np.testing.assert_allclose(r2, r3, rtol=2e-3)


class NeumannHelpers(unittest.TestCase):
    """Module-level helpers for the non-subwavelength leading-order code."""

    def test_jn_prime_zeros_known_values(self):
        # alpha_{1,0} ~ 2.0816, alpha_{1,1} ~ 5.9404, alpha_{1,2} ~ 9.2058
        z1 = _spherical_jn_prime_zeros(1, 3)
        np.testing.assert_allclose(
            z1, [2.0815759778, 5.9403699906, 9.2058401429], rtol=1e-8,
        )
        # alpha_{0,1} ~ 4.4934 (smallest positive root of tan(x) = x)
        z0 = _spherical_jn_prime_zeros(0, 1)
        np.testing.assert_allclose(z0, [4.4934094579], rtol=1e-8)
        # alpha_{2,0} ~ 3.3421 (first root of j_2'(x))
        z2 = _spherical_jn_prime_zeros(2, 1)
        np.testing.assert_allclose(z2, [3.3420936574], rtol=1e-8)

    def test_kappa_real_and_finite(self):
        # For l >= 1, alpha^2 > l(l+1), so the kappa formula is well-defined.
        k1 = _neumann_trace_kappa(1, 2.0815759778, 1.0)
        self.assertTrue(np.isfinite(k1))
        self.assertIsInstance(k1, float)
        # l=2, alpha=3.3421: alpha^2 ~ 11.17 > l(l+1)=6.
        k2 = _neumann_trace_kappa(2, 3.3420936574, 1.0)
        self.assertTrue(np.isfinite(k2))


class NonSubwavelengthApprox(unittest.TestCase):
    """Validate omega_n = omega_0 + lambda_n(omega_0) + O(delta^2) (Theorem 3.7).

    The frequency-dependent capacitance matrix C(omega_0) is assembled by
    discretising eq. (3.8) in the spherical-harmonic multipole basis. Its
    eigenvalues predict the leading-order non-subwavelength resonance shifts
    at a Neumann eigenfrequency omega_0 of -Laplace.
    """

    def test_neumann_enumeration_skips_constant_mode(self):
        """Only POSITIVE alpha_{l,n} are enumerated (alpha=0 constant mode
        belongs to the classical subwavelength regime and is excluded)."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        recs = cn.compute_neumann_eigenvalues(l_max=1, n_max=1)
        # Expect exactly 2 records: (l=0,n=0,alpha~4.493) and (l=1,n=0,alpha~2.082).
        self.assertEqual(len(recs), 2)
        alphas = sorted(r['alpha'] for r in recs)
        np.testing.assert_allclose(alphas, [2.0815759778, 4.4934094579], rtol=1e-8)

    def test_single_sphere_l1_dipole_matches_exact(self):
        """N=1: l=1,n=0 cluster is 3-fold degenerate; leading-order shifts
        match the exact solver (for small delta)."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        clusters = cn.find_omega0_candidates(l_max=1, n_max=1)
        cluster = next(c for c in clusters if c['size'] == 3)
        omega_0 = cluster['omega_0']

        delta = 1e-3
        N_multipole = 3

        C = cn.get_frequency_dependent_capacitance(
            omega_0, cluster['index_set'], cluster['kappa'], N_multipole, delta,
        )
        self.assertEqual(C.shape, (3, 3))

        omega_lead = cn.compute_nonsubwavelength_resonances(
            omega_0, cluster['index_set'], cluster['kappa'], N_multipole, delta,
        )
        roots = cn.compute_resonances(
            x0=omega_0 - 0.05 + 0j, N_roots=3, delta=delta,
            N_multipole=N_multipole, perturbation=1e-3,
        )

        lead_sorted = np.sort_complex(omega_lead)
        exact_sorted = np.sort_complex(roots)

        # Real parts: O(delta^2) absolute error => O(delta) relative to shift
        # (which is O(delta) itself). We compare as |re(lead) - re(exact)| / re(exact).
        np.testing.assert_allclose(
            lead_sorted.real, exact_sorted.real, rtol=1e-4,
            err_msg=f"Re(lead)={lead_sorted.real}, Re(exact)={exact_sorted.real}",
        )
        # Imag parts: O(delta) absolute (the radiative damping); tolerate
        # O(delta) absolute error => use atol = delta.
        np.testing.assert_allclose(
            lead_sorted.imag, exact_sorted.imag, atol=2 * delta,
            err_msg=f"Im(lead)={lead_sorted.imag}, Im(exact)={exact_sorted.imag}",
        )

    def test_two_sphere_chain_l1_matches_exact(self):
        """N=2 colinear chain at the l=1 Neumann cluster (6 degenerate modes)."""
        centers, radii, v_in = _chain(N=2, spacing=3.0)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        clusters = cn.find_omega0_candidates(l_max=1, n_max=1)
        cluster = next(c for c in clusters if c['size'] == 6)
        omega_0 = cluster['omega_0']

        delta = 1e-3
        N_multipole = 3

        C = cn.get_frequency_dependent_capacitance(
            omega_0, cluster['index_set'], cluster['kappa'], N_multipole, delta,
        )
        self.assertEqual(C.shape, (6, 6))

        omega_lead = cn.compute_nonsubwavelength_resonances(
            omega_0, cluster['index_set'], cluster['kappa'], N_multipole, delta,
        )
        roots = cn.compute_resonances(
            x0=omega_0 - 0.05 + 0j, N_roots=6, delta=delta,
            N_multipole=N_multipole, perturbation=1e-3,
        )

        lead_sorted = np.sort_complex(omega_lead)
        exact_sorted = np.sort_complex(roots)
        np.testing.assert_allclose(
            lead_sorted.real, exact_sorted.real, rtol=1e-4,
            err_msg=f"Re(lead)={lead_sorted.real}, Re(exact)={exact_sorted.real}",
        )
        np.testing.assert_allclose(
            lead_sorted.imag, exact_sorted.imag, atol=2 * delta,
            err_msg=f"Im(lead)={lead_sorted.imag}, Im(exact)={exact_sorted.imag}",
        )

    def test_delta_scaling_of_shift(self):
        """The shift |omega_n - omega_0| should scale linearly with delta."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        cluster = next(c for c in cn.find_omega0_candidates(l_max=1, n_max=1)
                       if c['size'] == 3)
        omega_0 = cluster['omega_0']

        def max_shift(delta):
            omega_lead = cn.compute_nonsubwavelength_resonances(
                omega_0, cluster['index_set'], cluster['kappa'], 3, delta,
            )
            return np.max(np.abs(omega_lead - omega_0))

        s1 = max_shift(1e-3)
        s2 = max_shift(5e-4)
        self.assertAlmostEqual(s1 / s2, 2.0, delta=0.05)


class SpuriousFiltering(unittest.TestCase):
    """Filter for fictitious single-layer resonances at interior Dirichlet
    eigenfrequencies of each ball (zeros of j_l(kR))."""

    def test_candidate_set_matches_known_zeros(self):
        """For a single unit sphere with v=1, candidates are exactly the
        (n_max * (l_max+1)) positive zeros of j_l.

        We compare against closed-form l=0 zeros (n * pi) and tabulated l=1
        zeros (4.4934094579..., 7.7252518369...) which are standard values."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        got = cn.spurious_omega_candidates(l_max=1, n_max=2)
        expected = np.sort(np.array([
            np.pi, 2.0 * np.pi,           # zeros of j_0(x) = sin(x)/x
            4.493409457909064, 7.725251836937707,  # zeros of j_1
        ]))
        # The first l=0 zero should be pi.
        self.assertAlmostEqual(got[0], np.pi, places=8)
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_candidate_set_scales_with_radius_and_speed(self):
        """Candidates = v * beta_{l,n} / R_i per-resonator."""
        centers = np.array([[0.0, 0.0, 0.0]])
        cn = ContinuumFiniteSWP3D(centers=centers, radii=2.0, v_in=1.0)
        # Only l=0 => first zero is pi/R = pi/2
        got = cn.spurious_omega_candidates(l_max=0, n_max=1)
        self.assertAlmostEqual(float(got[0]), np.pi / 2.0, places=10)

    def test_filter_spurious_removes_pi_root_N1(self):
        """At delta=1e-10 on the unit sphere, omega=pi is a spurious root of
        det A. With filter_spurious=True the returned roots are all physical
        (within spurious_tol of no spurious candidate)."""
        from Utils.utils_general import find_roots_muller

        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        delta = 1e-10
        N_multipole = 2

        # Baseline (no filter): at least one root lands on pi
        raw = cn.compute_resonances(x0=3.0 + 0j, N_roots=4, delta=delta,
                                    N_multipole=N_multipole, perturbation=1e-3,
                                    filter_spurious=False)
        self.assertTrue(np.any(np.abs(raw.real - np.pi) < 1e-4),
                        msg=f"Expected pi among raw roots, got {raw}")

        # Filtered: no root should land near any spurious candidate
        filtered = cn.compute_resonances(x0=3.0 + 0j, N_roots=4, delta=delta,
                                         N_multipole=N_multipole,
                                         perturbation=1e-3,
                                         filter_spurious=True,
                                         spurious_l_max=1, spurious_n_max=3)
        spurious = cn.spurious_omega_candidates(l_max=1, n_max=3)
        for r in filtered:
            dist = float(np.min(np.abs(spurious - r)))
            self.assertGreater(dist, 1e-4,
                               msg=f"Filtered root {r} is within {dist:.2e} "
                                   f"of spurious set {spurious}")

    def test_filter_preserves_physical_cluster_N1(self):
        """With filter enabled, Muller starting near the l=1,n=0 Neumann level
        recovers the triply-degenerate physical cluster near alpha_{1,0}."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        delta = 1e-4
        N_multipole = 3
        alpha_10 = 2.0815759778178706

        roots = cn.compute_resonances(x0=(alpha_10 - 0.05) + 0j, N_roots=3,
                                      delta=delta, N_multipole=N_multipole,
                                      perturbation=1e-3,
                                      filter_spurious=True)
        # All three should be near alpha_{1,0}
        self.assertTrue(np.all(np.abs(roots.real - alpha_10) < 5e-3),
                        msg=f"Roots {roots} should cluster near alpha_10={alpha_10}")

    def test_candidate_set_multiplicities_single_sphere(self):
        """Each (i, l, n) contributes 2 l + 1 to the multiplicity at
        omega = v * beta_{l,n} / R_i. For a single sphere no tuples coincide,
        so multiplicities equal 2 l + 1 per candidate."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        omegas, mults = cn.spurious_omega_candidates(
            l_max=1, n_max=2, with_multiplicities=True)
        # Expected order (sorted): pi (l=0), beta_{1,0}, 2pi (l=0), beta_{1,1}
        expected_omegas = np.array([
            np.pi, 4.493409457909064, 2.0 * np.pi, 7.725251836937707,
        ])
        expected_mults = np.array([1, 3, 1, 3], dtype=int)
        np.testing.assert_allclose(omegas, expected_omegas, atol=1e-9)
        np.testing.assert_array_equal(mults, expected_mults)

    def test_candidate_set_multiplicities_accumulate(self):
        """Resonators sharing a radius contribute their (2 l + 1) multiplicities
        at coincident omega values."""
        # Two resonators with equal radii -> candidates coincide per (l, n).
        centers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        radii = np.array([1.0, 1.0])
        v_in = np.array([1.0, 1.0])
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        omegas, mults = cn.spurious_omega_candidates(
            l_max=1, n_max=1, with_multiplicities=True)
        # Expected: pi with mult 2 (two l=0 tuples), beta_{1,0} with mult 6.
        expected_omegas = np.array([np.pi, 4.493409457909064])
        expected_mults = np.array([2, 6], dtype=int)
        np.testing.assert_allclose(omegas, expected_omegas, atol=1e-9)
        np.testing.assert_array_equal(mults, expected_mults)

    def test_pre_deflation_uses_correct_multiplicity(self):
        """The filter must divide det A by (w - s) ** m for each spurious s,
        not by (w - s) ** 1. We verify this directly: at omega = pi (mult 1)
        and omega = 2 pi (mult 1), dividing by (w - s) yields a finite non-zero
        limit. The multiplicity API should report these as m=1."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        omegas, mults = cn.spurious_omega_candidates(
            l_max=0, n_max=2, with_multiplicities=True)
        # l=0 contributes 2 l + 1 = 1 per (i, l, n).
        np.testing.assert_array_equal(mults, np.array([1, 1], dtype=int))

        # Direct deflation check: det A / (w - pi) has a finite non-zero limit
        # at w -> pi, since the zero at pi has multiplicity exactly 1.
        delta = 1e-10
        N_multipole = 2
        eps = 1e-6
        val = cn.characteristic_determinant(np.pi + eps + 0j, delta, N_multipole)
        deflated = val / eps
        # Compare to a slightly different eps to confirm stability (not -> 0).
        val2 = cn.characteristic_determinant(np.pi + 2 * eps + 0j, delta,
                                             N_multipole)
        deflated2 = val2 / (2 * eps)
        # Both should be close (same finite limit) and non-zero.
        self.assertGreater(abs(deflated), 1e-20)
        np.testing.assert_allclose(deflated, deflated2, rtol=1e-3)

    def test_pre_deflation_higher_multiplicity_for_l1(self):
        """At omega = beta_{1,0} the multiplicity is at least 3 (from l=1,
        m in {-1,0,1} spurious columns). Dividing by (w - s) ** 1 leaves a
        zero of order >= 2 at s; dividing by (w - s) ** 3 removes the spurious
        part. (There is in fact an additional physical Neumann l=0 zero at
        the same point because j_0'(x) = -j_1(x), but that is not spurious.)

        We verify that the API reports m=3 for the l=1 spurious candidate."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        omegas, mults = cn.spurious_omega_candidates(
            l_max=1, n_max=1, with_multiplicities=True)
        beta_10 = 4.493409457909064
        # Order: [pi, beta_10]
        np.testing.assert_allclose(omegas, [np.pi, beta_10], atol=1e-9)
        np.testing.assert_array_equal(mults, np.array([1, 3], dtype=int))

    def test_filter_matches_no_filter_for_physical_roots(self):
        """Away from spurious frequencies the filter should not perturb
        physical roots (it only divides out zeros of the pre-deflation
        polynomial, which are far from the physical cluster)."""
        centers, radii, v_in = _chain(N=1)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=radii, v_in=v_in)
        delta = 1e-4
        N_multipole = 3
        alpha_10 = 2.0815759778178706

        raw = cn.compute_resonances(x0=(alpha_10 - 0.05) + 0j, N_roots=3,
                                    delta=delta, N_multipole=N_multipole,
                                    perturbation=1e-3, filter_spurious=False)
        flt = cn.compute_resonances(x0=(alpha_10 - 0.05) + 0j, N_roots=3,
                                    delta=delta, N_multipole=N_multipole,
                                    perturbation=1e-3, filter_spurious=True)
        raw_s = np.sort_complex(raw)
        flt_s = np.sort_complex(flt)
        np.testing.assert_allclose(raw_s.real, flt_s.real, rtol=1e-6, atol=1e-8)


def _complex_sh_symmetry_conjugation(index_set):
    """Build ``J`` with ``J[(l, m), (l', m')] = (-1)^m * delta_{l l'} * delta_{m, -m'}``.

    Applied within each (j, l, n) block; zero elsewhere. In the complex-SH
    basis of :meth:`get_frequency_dependent_capacitance`, the operator-level
    complex symmetry of Prop. 3.13 manifests as ``C^T = J C J`` (rather than
    the naive ``C^T = C``), because ``{Y_l^m}`` is orthonormal in the
    Hermitian L^2 pairing, not the symmetric one used in the proposition.
    """
    M = len(index_set)
    J = np.zeros((M, M), dtype=complex)
    i = 0
    while i < M:
        j, l, n, _m = index_set[i]
        block = 2 * l + 1
        for a in range(block):
            m_a = a - l
            # Partner index with m -> -m within the same (j, l, n) block:
            b = (-m_a) + l
            J[i + a, i + b] = (-1.0) ** m_a
        i += block
    return J


class ComplexSymmetryProp313(unittest.TestCase):
    """Proposition 3.13 of the paper: ``C(omega_0)^T = C(omega_0)``.

    The proof uses the non-Hermitian L^2 pairing
    ``<f, g> = int f g dsigma`` on the boundary, applied to the *real-valued*
    Neumann boundary traces ``g_alpha``. In a real-valued basis the claim
    manifests directly as matrix complex-symmetry.

    :meth:`ContinuumFiniteSWP3D.get_frequency_dependent_capacitance`, however,
    works in the *complex* spherical-harmonic basis ``{Y_l^m}`` with the
    *Hermitian* pairing. The Prop. 3.13 complex-symmetry therefore manifests
    in two equivalent ways:

        1. After conjugation with the real-SH unitary ``U``,
           ``U C U^H`` is complex-symmetric.
        2. In the complex-SH basis, ``C^T = J C J``, where
           ``J[(l, m), (l', m')] = (-1)^m * delta_{l l'} * delta_{m, -m'}``.

    Both are stated and enforced below, so a future refactor of the basis
    conventions in :mod:`Subwavelength3D.continuum` cannot silently break the
    operator-level symmetry claimed by Prop. 3.13.
    """

    def _build_C(self, centers, R=1.0, v_in=1.0, delta=1e-2, N_multipole=3):
        cn = ContinuumFiniteSWP3D(
            centers=np.asarray(centers, dtype=float),
            radii=R, v_in=v_in,
        )
        c0 = cn.find_omega0_candidates(l_max=1, n_max=1)[0]
        C = cn.get_frequency_dependent_capacitance(
            c0['omega_0'], c0['index_set'], c0['kappa'],
            N_multipole=N_multipole, delta=delta,
        )
        return C, c0['index_set']

    def test_C_is_not_naively_symmetric_in_complex_sh_basis(self):
        """Document the convention: ``C`` is NOT ``C.T`` in the complex-SH basis.

        This guard fires if a future change silently switches the basis (e.g.
        to real SHs), which would make the rest of the library's tests and
        downstream consumers interpret the matrix differently. If that change
        is deliberate, update this test together with the convention.
        """
        # 2x2x2 cubic crystal: 8 identical resonators, l = 1 Neumann cluster.
        spacing = 5.0
        centers = np.array([[i * spacing, j * spacing, k * spacing]
                            for i in range(2) for j in range(2) for k in range(2)],
                           dtype=float)
        C, _ = self._build_C(centers)
        resid = np.linalg.norm(C - C.T) / np.linalg.norm(C)
        self.assertGreater(
            resid, 1e-3,
            msg="C appears complex-symmetric in the complex-SH basis — the "
                "library may have switched to a real-SH basis. Update this "
                "test and the convention in continuum.py if so."
        )

    @parameterized.expand([
        # Single resonator: Prop. 3.13 is trivial (C diagonal within l=1).
        [np.array([[0.0, 0.0, 0.0]])],
        # z-axis chain of two: couplings use mu = 0 only.
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])],
        # y-axis chain of two: couplings use all mu (tests J C J off-axis).
        [np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])],
        # Cube of eight: full 3D coupling.
        [np.array([[i * 5.0, j * 5.0, k * 5.0]
                   for i in range(2) for j in range(2) for k in range(2)],
                  dtype=float)],
    ])
    def test_complex_symmetry_after_real_sh_rotation(self, centers):
        """Prop. 3.13 in form 1: ``U C U^H`` is complex-symmetric."""
        C, index_set = self._build_C(centers)
        U = _complex_to_real_sh_unitary(index_set)
        # U is unitary.
        np.testing.assert_allclose(
            U @ U.conj().T, np.eye(len(index_set)), atol=1e-12,
        )
        C_real = U @ C @ U.conj().T
        resid = np.linalg.norm(C_real - C_real.T) / np.linalg.norm(C_real)
        self.assertLess(resid, 1e-10,
                        msg=f"U C U^H failed complex symmetry: resid = {resid:.2e}")

    @parameterized.expand([
        [np.array([[0.0, 0.0, 0.0]])],
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])],
        [np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])],
        [np.array([[i * 5.0, j * 5.0, k * 5.0]
                   for i in range(2) for j in range(2) for k in range(2)],
                  dtype=float)],
    ])
    def test_complex_symmetry_via_J_conjugation(self, centers):
        """Prop. 3.13 in form 2: ``C^T = J C J`` in the complex-SH basis.

        This is an exact algebraic identity (up to numerical noise) and is
        the most direct diagnostic of the operator-level complex symmetry in
        the basis the library actually uses.
        """
        C, index_set = self._build_C(centers)
        J = _complex_sh_symmetry_conjugation(index_set)
        # J is its own inverse (involution).
        np.testing.assert_allclose(
            J @ J, np.eye(len(index_set)), atol=1e-12,
        )
        resid = np.linalg.norm(C.T - J @ C @ J) / np.linalg.norm(C)
        self.assertLess(resid, 1e-10,
                        msg=f"C^T = J C J failed: resid = {resid:.2e}")

    def test_eigenvalues_invariant_under_real_sh_rotation(self):
        """Sanity: ``U`` is unitary so eigenvalues are basis-independent.

        This is an independent cross-check that downstream users of
        :meth:`compute_nonsubwavelength_resonances` are unaffected by the
        convention choice: the physical spectrum is the same.
        """
        spacing = 5.0
        centers = np.array([[i * spacing, j * spacing, k * spacing]
                            for i in range(2) for j in range(2) for k in range(2)],
                           dtype=float)
        C, index_set = self._build_C(centers)
        U = _complex_to_real_sh_unitary(index_set)
        C_real = U @ C @ U.conj().T
        eigs_c = np.sort_complex(np.linalg.eigvals(C))
        eigs_r = np.sort_complex(np.linalg.eigvals(C_real))
        np.testing.assert_allclose(eigs_c, eigs_r, atol=1e-10)

    def test_real_sh_basis_option_returns_complex_symmetric(self):
        """``get_frequency_dependent_capacitance(..., real_sh_basis=True)``
        must return a complex-symmetric matrix matching ``U C U^H``.

        This locks in the public API: callers asking for the Prop. 3.13
        form of the matrix get it directly, without needing to know the
        unitary ``U`` or the internal basis convention.
        """
        spacing = 5.0
        centers = np.array([[i * spacing, j * spacing, k * spacing]
                            for i in range(2) for j in range(2) for k in range(2)],
                           dtype=float)
        cn = ContinuumFiniteSWP3D(centers=centers, radii=1.0, v_in=1.0)
        c0 = cn.find_omega0_candidates(l_max=1, n_max=1)[0]

        C_complex = cn.get_frequency_dependent_capacitance(
            c0['omega_0'], c0['index_set'], c0['kappa'],
            N_multipole=3, delta=1e-2,
        )
        C_real = cn.get_frequency_dependent_capacitance(
            c0['omega_0'], c0['index_set'], c0['kappa'],
            N_multipole=3, delta=1e-2,
            real_sh_basis=True,
        )

        # Option agrees with manual U C U^H transform.
        U = _complex_to_real_sh_unitary(c0['index_set'])
        np.testing.assert_allclose(C_real, U @ C_complex @ U.conj().T, atol=1e-12)

        # Returned matrix is complex-symmetric (Prop. 3.13).
        resid = np.linalg.norm(C_real - C_real.T) / np.linalg.norm(C_real)
        self.assertLess(resid, 1e-10)

        # Same eigenvalues as the default-basis matrix.
        eigs_default = np.sort_complex(np.linalg.eigvals(C_complex))
        eigs_real = np.sort_complex(np.linalg.eigvals(C_real))
        np.testing.assert_allclose(eigs_default, eigs_real, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
