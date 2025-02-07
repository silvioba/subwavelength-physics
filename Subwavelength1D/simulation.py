import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors

from tqdm import tqdm
from typing import Callable, Optional
from Subwavelength1D.classic import ClassicFiniteSWP1D


class ClassicFiniteSWP1DSolverFlux:
    """
    FDTD solver for the 1D acoustic wave equation with piecewise kappa(x), rho(x):
        1/kappa(x) * d^2u/dt^2 = d/dx( 1/rho(x) * d/dx(u) ) + F_src(x,t),

    using a second-order flux-difference scheme that incorporates jumps in 1/rho.
    It supports:

      - Arbitrary left/right boundary conditions ('dirichlet' or 'absorbing')
      - Optional point-source injection in the interior: source(t) at x= x_src
      - Customizable zero or user-provided initial conditions

    Variables:
      alpha[i] = 1 / kappa(x_i)
      beta[i]  = 1 / rho(x_i)
    """

    def __init__(
        self,
        problem: ClassicFiniteSWP1D,
        Nx: int,
        T: float,
        cfl_safety: float = 0.9,
        x_padding: float | tuple = (1.0, 1.0),
        bc_left: str = "dirichlet",
        bc_right: str = "dirichlet",
        f: Optional[Callable[[float], float]] = None,
        g: Optional[Callable[[float], float]] = None,
        point_source_loc: Optional[float] = None,
        point_source_func: Optional[Callable[[float], float]] = None,
        point_source_width: Optional[float] = 0.1,
    ):
        """
        Args:
            problem (ClassicFiniteSWP1D): Subwavelength problem with geometry, wave speeds, etc.
            Nx (int): number of spatial grid points
            T (float): final time
            cfl_safety (float): safety factor for stable time step
            x_padding (float or tuple): domain extension on left/right, e.g. 1.0 => both sides = 1.0
            bc_left (str): 'dirichlet' or 'absorbing' for x=left boundary
            bc_right (str): 'dirichlet' or 'absorbing' for x=right boundary
            f (callable): initial condition u(0,x). Defaults to 0 if None.
            g (callable): initial condition d/dt u(0,x). Defaults to 0 if None.
            point_source_loc (float): location of point source. If None, no source.
            point_source_func (callable): forcing function source(t). If None => no source.

        Note:
            If you specify a point_source_loc and point_source_func,
            a forcing is added to the PDE at the grid node nearest to that x.
        """
        self.problem = problem
        self.Nx = Nx
        self.T = T
        self.cfl_safety = cfl_safety

        # Boundary conditions
        self.bc_left = bc_left.lower()
        self.bc_right = bc_right.lower()
        if self.bc_left not in ["dirichlet", "absorbing"]:
            raise ValueError("bc_left must be 'dirichlet' or 'absorbing'")
        if self.bc_right not in ["dirichlet", "absorbing"]:
            raise ValueError("bc_right must be 'dirichlet' or 'absorbing'")

        # User initial conditions or default zero
        self.f = (f if f else (lambda x: 0.0))
        self.g = (g if g else (lambda x: 0.0))

        # Domain geometry
        self.L = problem.L

        # Convert x_padding to (pad_left, pad_right) if single float
        if isinstance(x_padding, (int, float)):
            self.x_padding = (x_padding, x_padding)
        else:
            self.x_padding = x_padding

        # Spatial grid: [ -pad_left,  L + pad_right ]
        self.xx = np.linspace(-self.x_padding[0],
                              self.L + self.x_padding[1], Nx)
        self.dx = self.xx[1] - self.xx[0]

        # Build arrays alpha=1/kappa, beta=1/rho
        self.alpha = np.zeros(Nx, dtype=float)
        self.beta = np.zeros(Nx, dtype=float)
        self._compute_alpha_beta_arrays()

        # Compute stable dt from max wave speed c= sqrt(kappa/rho)= sqrt(1/alpha*beta)
        c_vals = np.sqrt((1.0/self.alpha) * self.beta)
        cmax = np.max(c_vals)
        self.dt = self.cfl_safety * self.dx / cmax

        # Number of time steps
        self.nt = int(np.ceil(self.T / self.dt))
        self.tt = np.arange(self.nt+1)*self.dt  # times

        # Prepare solution array
        self.u = np.zeros((self.nt+1, Nx), dtype=float)

        # Optional point source
        self.point_source_loc = point_source_loc
        self.point_source_func = point_source_func
        self.point_source_width = point_source_width
        self.i_source = None
        if (self.point_source_loc is not None) and (self.point_source_func is not None):
            # find grid index i near x= point_source_loc
            self.i_source = np.argmin(np.abs(self.xx - self.point_source_loc))

        # Initialize time
        self._initialize_time_step()

    def _compute_alpha_beta_arrays(self):
        """
        Fill self.alpha[i], self.beta[i] for the piecewise domain:
          - inside resonators => kappa_in= delta*(v_in[i]^2), rho_in=delta
          - outside => kappa_out=v_out^2, rho_out=1
        plus the padding => outside region
        """
        def inside_resonator(i):
            kappa_in_i = self.problem.delta*(self.problem.v_in[i]**2)
            return (1.0/kappa_in_i, 1.0/self.problem.delta)

        def outside():
            # alpha=1/kappa_out, beta=1/1=1
            return (1.0/(self.problem.v_out**2), 1.0)

        # We have intervals from xi= [0, l1, l1+s1, ...], up to L
        xi = self.problem.xi
        intervals = []
        N = self.problem.N

        for i in range(N):
            start_r, end_r = xi[2*i], xi[2*i+1]  # resonator
            alpha_r, beta_r = inside_resonator(i)
            intervals.append((start_r, end_r, alpha_r, beta_r))

            if (2*i+2) < len(xi):  # spacing
                start_o, end_o = xi[2*i+1], xi[2*i+2]
                alpha_o, beta_o = outside()
                intervals.append((start_o, end_o, alpha_o, beta_o))

        # Padding => outside
        alpha_o, beta_o = outside()
        if (self.x_padding[0] > 1e-14):
            intervals.append((-self.x_padding[0], 0.0, alpha_o, beta_o))
        if (self.x_padding[1] > 1e-14):
            intervals.append(
                (self.L, self.L+self.x_padding[1], alpha_o, beta_o))

        intervals.sort(key=lambda x: x[0])  # sort by start

        for (start, end, a_val, b_val) in intervals:
            s_clamped = max(start, -self.x_padding[0])
            e_clamped = min(end,   self.L + self.x_padding[1])
            mask = (self.xx >= s_clamped) & (self.xx < e_clamped)
            # If last boundary, include equality
            if np.isclose(e_clamped, self.L + self.x_padding[1]):
                mask = (self.xx >= s_clamped) & (self.xx <= e_clamped)

            self.alpha[mask] = a_val
            self.beta[mask] = b_val

    def _initialize_time_step(self):
        """
        Set initial conditions:
           u^0_i = f(x_i),   u^1_i = u^0_i + dt*g(x_i)
        """
        # n=0
        for i in range(self.Nx):
            self.u[0, i] = self.f(self.xx[i])

        # n=1
        for i in range(self.Nx):
            self.u[1, i] = self.u[0, i] + self.dt*self.g(self.xx[i])

        # Apply BC at n=0, n=1
        self._apply_bc(0)
        self._apply_bc(1)

    def _apply_bc(self, n):
        """
        Enforce boundary conditions at time step n for left and right edges.
        """
        # Left boundary
        if self.bc_left == "dirichlet":
            self.u[n, 0] = 0.0
        elif self.bc_left == "absorbing":
            # For an early time step, we might just match neighbor
            if n < 2:
                self.u[n, 0] = self.u[n, 1]
        else:
            raise NotImplementedError(f"Unknown bc_left={self.bc_left}")

        # Right boundary
        if self.bc_right == "dirichlet":
            self.u[n, -1] = 0.0
        elif self.bc_right == "absorbing":
            if n < 2:
                self.u[n, -1] = self.u[n, -2]
        else:
            raise NotImplementedError(f"Unknown bc_right={self.bc_right}")

    def solve(self) -> np.ndarray:
        """
        Main time-stepping loop (flux differencing in space + point-source).
        Returns:
            np.ndarray: shape (nt+1, Nx) for the solution u(t_n, x_i).
        """
        # For code cleanliness, define a small function for harmonic-average of beta
        def beta_harm(bi, bip1):
            if (bi > 0.0 and bip1 > 0.0):
                return 2.0*bi*bip1/(bi+bip1)
            return 0.0

        # Time stepping
        for n in tqdm(range(1, self.nt)):
            un = self.u[n, :]
            unm = self.u[n-1, :]
            unp = np.zeros_like(un)

            # 1) Compute fluxes at i+1/2
            flux = np.zeros(self.Nx+1, dtype=float)
            for i in range(self.Nx-1):
                bh = beta_harm(self.beta[i], self.beta[i+1])
                flux[i+1] = bh*(un[i+1]-un[i])/self.dx

            # 2) Update interior points
            for i in range(1, self.Nx-1):
                divF = (flux[i+1] - flux[i])/self.dx

                # Optional point-source forcing at i_source
                forcing_term = 0.0
                if (self.i_source is not None) and np.abs(self.xx[i]-self.point_source_loc) < self.point_source_width:
                    t_n = n*self.dt
                    forcing_term = self.point_source_func(t_n)

                unp[i] = (2.0*un[i] - unm[i]
                          + (self.dt**2/self.alpha[i])*(divF + forcing_term))

            # 3) Boundary conditions at new time step (n+1)
            #    We'll do them after computing the interior
            #    so we can directly set unp[0], unp[-1].
            if self.bc_left == "dirichlet":
                unp[0] = 0.0
            elif self.bc_left == "absorbing":
                cL = np.sqrt(
                    (1.0/self.alpha[0])*self.beta[0]) if self.alpha[0] > 1e-14 else 0.0
                rL = cL*self.dt/self.dx if cL > 1e-14 else 0.0
                unp[0] = (
                    un[1]
                    + ((rL-1)/(rL+1) if (rL+1) != 0 else 0.0)*(unp[1]-un[0])
                )

            if self.bc_right == "dirichlet":
                unp[-1] = 0.0
            elif self.bc_right == "absorbing":
                cR = np.sqrt(
                    (1.0/self.alpha[-1])*self.beta[-1]) if self.alpha[-1] > 1e-14 else 0.0
                rR = cR*self.dt/self.dx if cR > 1e-14 else 0.0
                unp[-1] = (
                    un[-2]
                    + ((rR-1)/(rR+1) if (rR+1) != 0 else 0.0)*(unp[-2]-un[-1])
                )

            self.u[n+1, :] = unp

        return self.u


def plot_wave_solution_heatmap(t_array, x_array, u, wp):
    """
    Plots the solution u(t,x) in a spacetime heatmap:
      - Horizontal axis: x
      - Vertical axis: t
    """
    # t_array has shape (Nt+1,)
    # x_array has shape (Nx+1,)
    # u has shape (Nt+1, Nx+1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # imshow expects the image first index to go along vertical axis,
    # so axis=0 in u is "time", axis=1 is "space":

    for xil, xir in wp.xiCol:
        ax.axvspan(xil, xir, alpha=0.3, color="grey")

    norm = colors.TwoSlopeNorm(vmin=np.min(
        u)-1e-1, vcenter=0, vmax=np.max(u)+1e-1)

    im = ax.imshow(
        u,                              # shape = (Nt+1, Nx+1)
        extent=[x_array[0], x_array[-1], t_array[0], t_array[-1]],
        origin="lower",                 # place t=0 at the bottom
        aspect="auto",                  # stretch the image to fill axes
        cmap="bwr",                      # or choose your preferred colormap
        norm=norm
    )

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("u(t, x)")

    # Axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Wave Solution Heatmap")

    plt.show()
