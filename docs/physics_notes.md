# Physics and numerical notes

This is a Markdown document with LaTeX equations. It uses `$...$` for inline
mathematics and `$$...$$` for display mathematics. The display delimiters are
always unindented and placed on their own lines for compatibility with common
MathJax and KaTeX Markdown viewers.

## Coordinates and metric

The code uses units with $c=1$ and metric signature $(-,+,+,+)$. The
coordinates are laboratory time $t$, proper radial coordinate $l$, polar angle
$\theta$, and azimuth $\phi$.

The generalized Ellis-Bronnikov metric is

$$
ds^2=-dt^2+dl^2+R(l)^2\left(d\theta^2+\sin^2\theta\,d\phi^2\right).
$$

The areal-radius function is

$$
R(l)=\left(b_0^m+l^m\right)^{1/m}.
$$

Here, $b_0$ is the throat radius. The code requires $m$ to be an even integer
with $m\geq 2$, so $R(l)$ is real and smooth on both sides of the throat. The
case $m=2$ is the ordinary Ellis-Bronnikov wormhole.

The SciML experiments in this repository fix
$b_0=1$, $m=2$, $\omega=1$, $\alpha=-2$, and $\theta=\pi/6$.
Other parameter values below are equation checks or illustrative physics
examples and must not be confused with the learned-model dataset.

## Rotating spiral

The particle is constrained to a conical Archimedean spiral:

$$
\theta=\mathrm{constant},
\qquad
\Phi(t,l)=\alpha l+\omega t.
$$

The radial velocity and effective angular velocity are

$$
v=\frac{dl}{dt},
\qquad
\Omega=\frac{d\Phi}{dt}=\omega+\alpha v.
$$

Substitution into the metric leaves a two-dimensional metric in $(t,l)$:

$$
g_{ab}=
\begin{pmatrix}
-1+q\omega^2 & q\omega\alpha \\
q\omega\alpha & 1+q\alpha^2
\end{pmatrix},
\qquad
q=R(l)^2\sin^2\theta.
$$

Its determinant is

$$
\det g=-\left[1+q\left(\alpha^2-\omega^2\right)\right].
$$

## Physical velocity constraint

The total three-speed measured using laboratory time is

$$
v_{\mathrm{tot}}^2
=v^2+R(l)^2\sin^2\theta\left(\omega+\alpha v\right)^2.
$$

A massive particle must satisfy

$$
v_{\mathrm{tot}}^2<1.
$$

The two roots of $v_{\mathrm{tot}}^2=1$ form the upper and lower boundaries in
the phase-space figures. The code computes these roots in `velocity_bounds`.

For the force-free escaping regime, $|\omega|<|\alpha|$. The terminal radial
velocity is

$$
v_\infty=-\frac{\omega}{\alpha},
$$

and $\Omega\rightarrow 0$.

## Conserved energy

The Lorentz factor and conserved specific energy are

$$
\begin{aligned}
\gamma
&=\left(-g_{00}-2g_{01}v-g_{11}v^2\right)^{-1/2}, \\
E
&=-\gamma\left(g_{00}+g_{01}v\right).
\end{aligned}
$$

Solving these equations for $v$ gives two velocity branches. The function
`velocity_from_energy` evaluates either branch, while
`energy_branch_for_state` selects the branch passing through a specified
initial state.

## Radial equation

The integrated equation is

$$
\begin{aligned}
\frac{d^2l}{dt^2}
={}&-\frac{
\sin^2\theta\,(\omega+\alpha v)
(b_0^m+l^m)^{2/m-1}l^{m-1}
}{
1+\sin^2\theta(\alpha^2-\omega^2)R(l)^2
} \\
&\times\left[
\alpha v-\omega+2\omega v^2
+\sin^2\theta(\omega+\alpha v)^2\omega R(l)^2
\right].
\end{aligned}
$$

The function `radial_acceleration` contains this expression. The function
`radial_system` turns the second-order equation into two first-order equations:

$$
\frac{dl}{dt}=v,
\qquad
\frac{dv}{dt}=a(l,v).
$$

SciPy integrates this two-component system.

## Source-file map

| File | Main calculation |
|---|---|
| `src/wormhole_sciml/geometry.py` | Radius, metric, metric derivative, and determinant |
| `src/wormhole_sciml/dynamics.py` | Velocity constraint, energy, acceleration, and ODE right-hand side |
| `src/wormhole_sciml/integrate.py` | Physical and continued radial-flow integration with SciPy |
| `src/wormhole_sciml/observables.py` | Terminal velocity, azimuth, and trajectory coordinates |
| `scripts/make_figures.py` | Time plots, phase curves, vector fields, and 3D trajectories |
| `scripts/run_physics_gate.py` | Fixed-experiment physical domain, step-size, and reference-suite checks |

The original notebook and MATLAB files remain in the private provenance
archive. They are not required to run the corrected public implementation.

## Numerical method

The default integrator is SciPy's `DOP853`, an adaptive high-order explicit
Runge-Kutta method. The equations are smooth and non-stiff for the parameter
sets used here. The default tolerances in `integrate_trajectory` are
`rtol=1e-10` and `atol=1e-12`; scripts can override them.

The integration stops if a numerical trajectory reaches
$v_{\mathrm{tot}}^2=1$. An inadmissible initial state is rejected before the
solver starts.

`integrate_vector_field_continuation` is a separate numerical entry point for
the mathematical continuation of the same validated radial vector field. It
accepts finite states with $C=1-v_{\mathrm{tot}}^2\leq0$ and deliberately applies
no timelike or null-boundary semantics. These solves are diagnostic
continuations for controlled SciML studies, not physical particle
trajectories, and their call path does not evaluate the Lorentz factor or
conserved energy.

## Important source issues

### Missing factors in the printed paper

Paper equations (10) and (11) are missing factors of two. Direct reduction of
the coordinate-time geodesic equation gives

$$
\begin{aligned}
A_2&=2g_{01}g'_{01}+2g_{11}g'_{00}-g_{00}g'_{11}, \\
A_3&=2g_{11}g'_{01}-g_{01}g'_{11}.
\end{aligned}
$$

These corrected coefficients agree with the paper's final acceleration
equation and with the trajectory equations in the legacy files.

### Phase-boundary exponent

`WHEllisGen.m` uses exponent `2/m` in its phase-boundary expression. The
correct exponent for $l$ is `1/m`.

### Runge-Kutta stage

The MATLAB fourth Runge-Kutta velocity stage uses `dv2`; classical RK4 requires
`dv3`.

### Saved inadmissible state

The saved MATLAB initial state $(l_0,v_0)=(0.5,0)$ has
$v_{\mathrm{tot}}^2=1.25$ and is not physical for its saved parameters.

### Meaning of the 3D coordinates

The old 3D plots use signed proper coordinate $l$ as a Euclidean plotting
radius. They are useful coordinate pictures, but they are not isometric
embeddings of a wormhole spatial slice. The plotting code labels this
convention explicitly.

## Quantitative checks

These values are regenerated by `scripts/validate_baseline.py` under
`output/`; generated validation files are not committed.

- The closed acceleration and independent metric-derived acceleration agree to
  about $4\times 10^{-14}$.
- The analytic energy branch and integrated ODE velocity agree to about
  $3.2\times 10^{-11}$.
- Tightening solver tolerances changes sampled trajectories by less than
  $5.1\times 10^{-11}$.
- The through-going state $(l_0,v_0)=(-30,0.827)$ crosses the throat near
  $t=34.5$ and remains timelike.
