"""
Active Cahn–Hilliard–Darcy model: droplet fingering instability
===============================================================

Phase-field model of an active growing droplet (φ = +1) expanding into
a passive surrounding (φ = −1).  Fingering is controlled by the viscosity
ratio  r_vis = η_passive / η_active  (analogous to the parameter f in
Bogdan & Savin, RSOS 2018, and to the traction parameter in Alert et al.,
PRL 2019).

    r_vis = 1   →  equal viscosity, stable circular growth  (no fingering)
    r_vis > 1   →  active fluid invades more-viscous passive  →  fingering

──────────────────────────────────────────────────────────────────────────
EQUATIONS
──────────────────────────────────────────────────────────────────────────
Phase field         ∂φ/∂t = M ∇²μ  −  v · ∇φ
Chemical potential  μ = f′(φ) − ε² ∇²φ ,   f(φ) = ¼(φ²−1)²
Darcy velocity      v = −K(φ) ∇p ,          K(φ) = 1/η(φ)
Pressure            ∇·[K(φ) ∇p] = S(φ) ,   S = k_grow (1+φ)/2

η(φ) interpolates linearly:  η = 1 at φ = +1 (active),
                              η = r_vis at φ = −1 (passive).

──────────────────────────────────────────────────────────────────────────
NUMERICS
──────────────────────────────────────────────────────────────────────────
•  Pseudo-spectral (2-D FFT), periodic domain
•  Stabilised semi-implicit time-stepping for the Cahn–Hilliard part
   (Eyre-type splitting; unconditionally stable for C_stab ≥ max|f″| = 2)
•  Picard iteration for the variable-permeability Poisson equation

──────────────────────────────────────────────────────────────────────────
SUGGESTED STUDENT WORKFLOW
──────────────────────────────────────────────────────────────────────────
  Step 1  Pure CH (no velocity): run with k_grow = 0, r_vis = 1.
          Verify a perturbed ellipse relaxes back to a circle.

  Step 2  Add growth: set k_grow > 0, r_vis = 1.
          Observe that the droplet grows circularly; measure R(t) and
          confirm exponential growth  R ~ R0 exp(k_grow t / 2).

  Step 3  Turn on activity: increase r_vis above 1.
          Watch fingers develop.  Find the critical r_vis* where they
          first appear.

  Step 4  Run compare_activity([1, 2, 4, 8]) to make a comparison figure.

  Step 5  Count fingers at late time as a function of r_vis and compare
          with Fig. 3 / Fig. 5 in Bogdan & Savin (2018).

──────────────────────────────────────────────────────────────────────────
REFERENCES
──────────────────────────────────────────────────────────────────────────
  Bogdan & Savin, R. Soc. Open Sci. 5, 181579 (2018)
  Alert, Blanch-Mercader & Casademunt, PRL 122, 088104 (2019)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ══════════════════════════════════════════════════════════════════════════
# PARAMETERS  –  edit this block to explore different physical regimes
# ══════════════════════════════════════════════════════════════════════════

# ── Grid ──────────────────────────────────────────────────────────────────
N          = 128                    # grid points per side (must be a power of 2)
L          = 256.0                  # domain side length (arbitrary units)

# ── Physics ───────────────────────────────────────────────────────────────
eps        = 5.0                    # diffuse interface width  (keep ≥ 2 dx for resolution)
M          = 1.0                    # Cahn–Hilliard mobility
k_grow     = 6e-4                   # volumetric growth rate in the active phase
r_vis      = 4.0                    # viscosity ratio η_passive / η_active   ← KEY PARAMETER
                                    #   r_vis = 1  →  no fingering
                                    #   r_vis > 1  →  fingering (more fingers at higher r_vis)

# ── Initial droplet ───────────────────────────────────────────────────────
R0         = 50.0                   # initial radius
noise_amp  = 0.1                    # amplitude of random interface perturbations

# ── Numerical ─────────────────────────────────────────────────────────────
dt         = 0.05                   # time step
n_steps    = 35000                  # total simulation steps  (total time = n_steps × dt)
plot_every = int(n_steps / (4*2-1)) # steps between saved frames, 4 columns and 2 rows
n_p_iter   = 15                     # Picard iterations for the pressure solve (increase to ~25 for r_vis > 8)
C_stab     = 2.0                    # stabilisation constant  (must be ≥ max|f″(φ)| = 2)

# ══════════════════════════════════════════════════════════════════════════
# GRID  AND  WAVENUMBERS  (do not edit)
# ══════════════════════════════════════════════════════════════════════════
dx        = L / N
x1d       = np.linspace(0, L, N, endpoint=False)
X, Y      = np.meshgrid(x1d, x1d)

k1d       = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)   # angular wavenumbers
kx, ky    = np.meshgrid(k1d, k1d)
k2        = kx**2 + ky**2                             # |k|²
k4        = k2**2                                     # |k|⁴
k2_nz     = np.where(k2 == 0, 1.0, k2)               # safe denominator (k=0 handled separately)

# Precomputed denominator for the semi-implicit CH update (see time_step):
#   φ̂ⁿ⁺¹ = numerator / denom_CH
# The implicit (C_stab k² + ε² k⁴) part removes all stiffness from the CH equation.
denom_CH  = 1.0 + dt * M * (C_stab * k2 + eps**2 * k4)


# ══════════════════════════════════════════════════════════════════════════
# SPECTRAL OPERATORS
# ══════════════════════════════════════════════════════════════════════════

def fft2(u):
    return np.fft.fft2(u)

def ifft2r(u_hat):
    """Inverse FFT; returns real part (discards negligible imaginary residual)."""
    return np.real(np.fft.ifft2(u_hat))

def gradient(u_hat):
    """∇u = (∂ₓu, ∂ᵧu) from Fourier array  u_hat."""
    return ifft2r(1j * kx * u_hat), ifft2r(1j * ky * u_hat)

def divergence(fx, fy):
    """∇·(fx, fy) computed spectrally."""
    return ifft2r(1j * kx * fft2(fx) + 1j * ky * fft2(fy))


# ══════════════════════════════════════════════════════════════════════════
# PHYSICS
# ══════════════════════════════════════════════════════════════════════════

def permeability(phi):
    """
    K(φ) = 1 / η(φ),  with linear viscosity interpolation:

        φ = +1  (active phase)  →  η = 1         →  K = 1
        φ = −1  (passive phase) →  η = r_vis     →  K = 1 / r_vis

    The active phase is more permeable (less viscous) than the passive phase.
    When r_vis > 1 this contrast drives a Saffman–Taylor-type instability,
    analogous to the active-traction term in Bogdan & Savin (2018).
    """
    eta = 1.0 + (r_vis - 1.0) * (1.0 - phi) / 2.0
    return 1.0 / eta


def growth_source(phi):
    """
    S(φ) = k_grow · (1 + φ) / 2

    Equals k_grow in the active bulk (φ = +1) and zero outside (φ = −1).
    Drives radial expansion of the droplet via  ∇·v = S.
    """
    return k_grow * (1.0 + phi) / 2.0


def solve_pressure(phi, K):
    """
    Solve  ∇·[K(φ) ∇p] = S(φ)  iteratively (Picard / fixed-point).

    Strategy: split  K = K̄ + δK  where K̄ = mean(K).
    Each iteration solves a constant-coefficient Poisson problem:

        K̄ ∇²pⁿ⁺¹ = S  −  ∇·(δK ∇pⁿ)

    which is diagonal in Fourier space → exact spectral solve per iteration.

    Convergence is guaranteed when  ‖δK‖_∞ / K̄  < 1, which holds for
    moderate r_vis.  Increase n_p_iter for r_vis > 8.

    Gauge: mean pressure is set to zero (p̂[0,0] = 0).
    """
    S     = growth_source(phi)
    K_bar = float(np.mean(K))
    dK    = K - K_bar

    # Initial guess: homogeneous solve (δK = 0)
    S_hat         = fft2(S)
    S_hat[0, 0]   = 0.0
    p_hat         = -S_hat / (K_bar * k2_nz)
    p_hat[0, 0]   = 0.0

    for _ in range(n_p_iter):
        px, py        = gradient(p_hat)
        corr          = divergence(dK * px, dK * py)   # ∇·(δK ∇p)
        rhs_hat       = fft2(S - corr)
        rhs_hat[0, 0] = 0.0
        p_hat         = -rhs_hat / (K_bar * k2_nz)
        p_hat[0, 0]   = 0.0

    return p_hat


# ══════════════════════════════════════════════════════════════════════════
# TIME STEP
# ══════════════════════════════════════════════════════════════════════════

def time_step(phi):
    """
    Advance φ by one time step dt.

    Algorithm
    ---------
    1.  K(φ)  via permeability()
    2.  Pressure:  ∇·[K ∇p] = S(φ)    (iterative spectral solve)
    3.  Darcy velocity:  v = −K ∇p
    4.  Advection:  v·∇φ               (explicit)
    5.  Cahn–Hilliard step             (stabilised semi-implicit)

        φ̂ⁿ⁺¹ =  φ̂ⁿ  −  dt M k² (f̂′ⁿ − C φ̂ⁿ)  −  dt adv_hat
                ─────────────────────────────────────────────────────
                         1  +  dt M C k²  +  dt M ε² k⁴

        f′(φ) = φ³ − φ  (from double-well  f = ¼(φ²−1)²)

    Returns
    -------
    phi_new : ndarray   updated phase field
    vx, vy  : ndarray   Darcy velocity components (for diagnostics / plotting)
    """
    phi_hat    = fft2(phi)
    K          = permeability(phi)

    # ── 1 & 2.  Pressure ──────────────────────────────────────────────────
    p_hat      = solve_pressure(phi, K)

    # ── 3.  Darcy velocity ────────────────────────────────────────────────
    px, py     = gradient(p_hat)
    vx, vy     = -K * px, -K * py

    # ── 4.  Advection ─────────────────────────────────────────────────────
    phix, phiy = gradient(phi_hat)
    adv_hat    = fft2(vx * phix + vy * phiy)

    # ── 5.  Cahn–Hilliard (stabilised semi-implicit) ─────────────────────
    #
    #  Full equation in Fourier:
    #    ∂φ̂/∂t  =  −M k² [f̂′  +  ε² k² φ̂]  −  adv_hat
    #                   ↑ from ∇²μ = −k² μ̂ , μ̂ = f̂′ + ε²k²φ̂
    #
    #  Stabilisation: add ±C k² φ̂ to split into explicit concave + implicit convex
    #    explicit part:  −M k² (f̂′ − C φ̂)
    #    implicit part:  −M k² C φ̂  −  M ε² k⁴ φ̂  → moved to denominator
    #
    f_prime    = phi**3 - phi                        # f′(φ) = φ³ − φ
    fp_hat     = fft2(f_prime)

    numerator  = (phi_hat - dt * M * k2 * (fp_hat - C_stab * phi_hat) - dt * adv_hat)
    phi_hat_new = numerator / denom_CH

    phi_new    = ifft2r(phi_hat_new)
    phi_new    = np.clip(phi_new, -1.0, 1.0)        # keep in physical range
    return phi_new, vx, vy


# ══════════════════════════════════════════════════════════════════════════
# INITIAL CONDITION
# ══════════════════════════════════════════════════════════════════════════

def make_initial_phi(seed=42, aspect=1.4):   # aspect > 1 → ellipse
    """
    Elliptical droplet with a smooth tanh interface profile, plus small random
    perturbations localised at the interface to seed the instability.

    The noise amplitude is noise_amp; its spatial support is a Gaussian
    centred on φ = 0 (the interface).
    """
    rng = np.random.default_rng(seed)
    r   = np.sqrt(((X - L/2) / aspect)**2 + ((Y - L/2) * aspect)**2)
    phi = np.tanh((R0 - r) / (np.sqrt(2) * eps))
    mask = np.exp(-10.0 * phi**2)
    phi += noise_amp * rng.standard_normal((N, N)) * mask
    return np.clip(phi, -1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════

def measure_radius(phi):
    """Effective radius from the area of the active region  (φ > 0)."""
    area = np.sum(phi > 0) * dx**2
    return np.sqrt(area / np.pi)


def count_fingers(phi):
    """
    Crude finger count: number of local maxima of the radius r(θ) along
    the φ = 0 contour, using a simple peak-finding approach.
    """
    from scipy.ndimage import label
    # Identify interface pixels
    interface = np.abs(phi) < 0.3
    labeled, n_components = label(interface)
    if n_components == 0:
        return 0
    # Project onto angle to find lobes
    cx, cy = L / 2, L / 2
    angles  = np.arctan2(Y - cy, X - cx)
    n_bins  = 360
    edges   = np.linspace(-np.pi, np.pi, n_bins + 1)
    r_vals  = np.sqrt((X - cx)**2 + (Y - cy)**2)
    # mean radius in each angular bin where interface exists
    counts, _ = np.histogram(angles[interface], bins=edges)
    r_mean, _ = np.histogram(angles[interface], bins=edges,
                              weights=r_vals[interface])
    valid  = counts > 0
    r_prof = np.where(valid, r_mean / np.maximum(counts, 1), 0.0)
    # Count peaks (crude)
    peaks = 0
    for i in range(n_bins):
        prev = r_prof[(i - 1) % n_bins]
        curr = r_prof[i]
        nxt  = r_prof[(i + 1) % n_bins]
        if valid[i] and curr > prev and curr >= nxt and curr > 0:
            peaks += 1
    return peaks


# ══════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_simulation(r_vis_value=None, k_grow_value=None, verbose=True):
    """
    Run the full simulation.

    Parameters
    ----------
    r_vis_value : float or None
        Override the global r_vis if provided.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    frames : list of (phi_array, time) tuples
        Snapshots saved every plot_every steps.
    diagnostics : dict
        Time series of 't', 'R' (radius), 'n_fingers'.
    """
    global r_vis
    if r_vis_value is not None:
        r_vis = r_vis_value
    
    global k_grow
    if k_grow_value is not None:
        k_grow = k_grow_value

    phi    = make_initial_phi()
    frames = [(phi.copy(), 0.0)]
    diag   = {'t': [0.0], 'R': [measure_radius(phi)], 'n_fingers': [0]}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  r_vis = {r_vis:.2f}  |  k_grow = {k_grow:.4f}  |  ε = {eps:.1f}  |  N = {N}")
        print(f"{'─'*60}")
        print(f"  {'step':>6}  {'t':>8}  {'R':>7}  {'fingers':>8}")
        print(f"  {'':─>6}  {'':─>8}  {'':─>7}  {'':─>8}")

    for n in range(1, n_steps + 1):
        phi, vx, vy = time_step(phi)

        if n % plot_every == 0:
            t   = n * dt
            R   = measure_radius(phi)
            nf  = count_fingers(phi)
            frames.append((phi.copy(), t))
            diag['t'].append(t)
            diag['R'].append(R)
            diag['n_fingers'].append(nf)
            if verbose:
                print(f"  {n:>6d}  {t:>8.1f}  {R:>7.1f}  {nf:>8d}")

    return frames, diag


# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════

_xs = np.linspace(0, L, N)    # shared axis array for contour overlays


def _draw_phi(phi, ax, title='', colorbar=False):
    """Draw phase field + φ=0 contour on a given Axes object."""
    im = ax.imshow(phi, extent=[0, L, 0, L], origin='lower',
                   cmap='RdBu_r', vmin=-1, vmax=1, interpolation='bilinear')
    ax.contour(_xs, _xs, phi, levels=[0.0], colors='k', linewidths=1.4)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    if colorbar:
        plt.colorbar(im, ax=ax, label='φ  (+1 active, −1 passive)', fraction=0.046, pad=0.04)
    return im


# ── Snapshot grid ─────────────────────────────────────────────────────────

def plot_snapshots(frames, filename='snapshots.png', n_cols=4):
    """
    Tile all saved frames in a single figure and save as PNG.
    """
    nf    = len(frames)
    ncols = min(n_cols, nf)
    nrows = 2
    # Plot only ncols times nrows frames
    frames_to_plot = frames[:ncols*nrows]

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), constrained_layout=False)
    axes = np.array(axes).flatten()

    for i, (phi, t) in enumerate(frames_to_plot):
        _draw_phi(phi, axes[i], title=f't = {t:.0f}')
    for ax in axes[nf:]:
        ax.axis('off')

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-1, 1))
    fig.subplots_adjust(wspace=0.3, right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('φ', rotation='horizontal', fontsize=14)

    fig.suptitle('$r_{vis} =$' + str(r_vis) + ', $k_{grow} =$' + str(k_grow) + ', $\\epsilon =$' + str(eps), fontsize=14, y=0.98)
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved  →  {filename}')
    plt.show()


# ── Growth and finger diagnostics ─────────────────────────────────────────

def plot_diagnostics(diag, filename='diagnostics.png'):
    """
    Plot R(t) and finger count n(t) from a diagnostics dictionary.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    t  = diag['t']
    R  = diag['R']
    nf = diag['n_fingers']

    ax1.plot(t, R, 'o-', color='steelblue', fillstyle='none', lw=1.2, ms=6)
    # Overlay theoretical exponential: R ~ R0 exp(k_grow t / 2)
    t_arr = np.linspace(0, max(t), 200)
    ax1.plot(t_arr, R0 * np.exp(k_grow * t_arr / 2), '--', color='firebrick', lw=1.5, label=r'$R_0 e^{k\, t/2}$')
    ax1.set_xlabel('$Time,$ $t$')
    ax1.set_ylabel('$Droplet$ $radius,$ $R$')
    ax1.set_title('$Droplet$ $growth$')
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax2.plot(t, nf, 's-', color='darkorange', fillstyle='none', lw=1.2, ms=6)
    ax2.set_xlabel('$Time,$ $t$')
    ax2.set_ylabel('$Number$ $of$ $fingers$')
    ax2.set_title('$Finger$ $count$ $(crude$ $estimate)$')
    ax2.grid(alpha=0.2)

    fig.suptitle('$r_{vis} =$' + str(r_vis) + ', $k_{grow} =$' + str(k_grow) + ', $\\epsilon =$' + str(eps), fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved  →  {filename}')
    plt.show()


# ── Animated GIF ──────────────────────────────────────────────────────────

def make_gif(frames, filename='active_droplet.gif', fps=6):
    """
    Create and save an animated GIF of the simulation.
    Requires Pillow  (pip install Pillow).
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    phi0, t0 = frames[0]

    im   = ax.imshow(phi0, extent=[0, L, 0, L], origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    cont = [ax.contour(_xs, _xs, phi0, levels=[0], colors='k', linewidths=1.5)]
    ttl  = ax.set_title('$t =$' + str(t0), fontsize=14)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('φ', rotation='horizontal', fontsize=14)

    def _update(idx):
        phi, t = frames[idx]
        im.set_data(phi)
        cont[0].remove()
        cont[0] = ax.contour(_xs, _xs, phi, levels=[0], colors='k', linewidths=1.5)
        ttl.set_text('$r_{vis} =$' + str(r_vis) + ', $k_{grow} =$' + str(k_grow) + ', $\\epsilon =$' + str(eps) + '\n$t =$' + str(t))
        return [im]

    ani = FuncAnimation(fig, _update, frames=len(frames), interval=int(1000 / fps), blit=False)
    ani.save(filename, writer=PillowWriter(fps=fps))
    print(f'Saved  →  {filename}')
    plt.close()
    return ani


# ── Side-by-side comparison ───────────────────────────────────────────────

def compare_activity(r_vis_list=(1.0, 2.0, 4.0, 8.0),
                     filename='compare_rvis.png'):
    """
    Run the model for several values of r_vis and display the final
    phase fields side-by-side.  Saves a PNG comparison figure.

    Example
    -------
    >>> compare_activity([1.0, 2.0, 4.0, 8.0])
    """
    global r_vis
    fig, axes = plt.subplots(1, len(r_vis_list),
                             figsize=(4 * len(r_vis_list), 4.5))
    axes = list(np.array(axes).flatten())
    im_last = None

    for ax, rv in zip(axes, r_vis_list):
        frames, _ = run_simulation(r_vis_value=rv, verbose=True)
        phi_f, t_f = frames[-1]
        im_last = _draw_phi(phi_f, ax, title=f'r_vis = {rv:.1f}')

    fig.colorbar(im_last, ax=axes, label='φ  (+1 active, −1 passive)',
                 shrink=0.75)
    fig.suptitle(
        f'Fingering instability at different viscosity ratios'
        f'   (t = {t_f:.0f})',
        fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'\nSaved  →  {filename}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── STEP 1 & 2: single run (pure CH or with growth) ───────────────────
    # Set k_grow = 0 and r_vis = 1 first to verify CH relaxation.
    # Then k_grow > 0, r_vis = 1 to verify circular growth.

    frames, diag = run_simulation(r_vis_value=float(sys.argv[1]), k_grow_value=float(sys.argv[2]))

    filename = f'r_vis{r_vis:.1f}-k_grow{k_grow:.4f}'
    plot_snapshots(frames,  filename=f'snapshots_{filename}.png')
    plot_diagnostics(diag,  filename=f'diagnostics_{filename}.png')
    make_gif(frames,        filename=f'droplet_{filename}.gif')

    # ── STEP 4: compare different activity levels ─────────────────────────
    # Uncomment the line below (and comment out the block above if needed)

    # compare_activity([1.0, 2.0, 4.0, 8.0])
