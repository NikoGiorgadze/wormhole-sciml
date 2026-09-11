# Model and variable definitions

This document fixes the public vocabulary and the interfaces needed to read
the code and compact results.  Historical artifact names are translated in
the final section.

## Fixed physical setting

The machine-learning experiments use

$$
b_0=1,\qquad m=2,\qquad \omega=1,\qquad \alpha=-2,
\qquad \theta=\pi/6,
$$

with units in which \(c=1\).  The evolved physical state is position and
radial velocity, \((x,u)\).  The timelike condition gives position-dependent
velocity bounds \(u_-(x)<u<u_+(x)\).  Their midpoint and half-width are

$$
c(x)=\frac{u_-(x)+u_+(x)}{2},\qquad
d(x)=\frac{u_+(x)-u_-(x)}{2}.
$$

The transformed coordinate

$$
\xi=\frac{u-c(x)}{d(x)}
$$

maps every admissible state to \(-1<\xi<1\).  This is a physical coordinate
transformation, not a fitted normalization.  Standardization for neural
network training is applied separately using training-split statistics.

The symbol \(E_0\) is the exact conserved energy evaluated once from the
initial state and then held fixed.  The symbol \(u_{\rm th}\) denotes the
radial velocity at the throat and is used as an orbit-family label, not as a
network input.

## Local residual model

The retained local task uses the fixed step \(h=0.2\):

$$
(x,\xi,E_0)\longmapsto(\Delta x,\Delta\xi).
$$

The selected model is a fully connected \(3\to32\to32\to2\) network with two
`tanh` hidden layers and a linear output, for 1,250 trainable parameters.  It
was trained on 40,000 rows and validated on 8,000 independently generated
rows with seeds 101, 202, and 303.  Training used Adam at \(10^{-3}\), batch
size 512, no weight decay or scheduler, a 1,500-epoch ceiling, patience 40,
and equal two-component standardized mean-squared error.  Seed 101 is the
primary checkpoint because it had the best validation score among the three
fixed-energy runs.

The local model is intended for both one-step evaluation and recursive
composition.  Those are separate tests: small one-step error does not imply a
stable recursive trajectory.

## Direct finite-time model

The finite-time input is

$$
(x_0,\xi_0,E_0,s),
$$

where \(s\geq0\) is elapsed physical time.  The selected representation uses

$$
V_x=\frac{\Delta x}{s},\qquad
F_\xi=\frac{\Delta\xi}{g(s)},\qquad
g(s)=-5\operatorname{expm1}(-s/5).
$$

At inference,

$$
\widehat{\Delta x}=s\widehat V_x,
\qquad
\widehat{\Delta\xi}=g(s)\widehat F_\xi.
$$

Because both gates vanish at \(s=0\), the identity map is exact by
construction.  The public name is **time-rescaled finite-time model**.

The network is \(4\to64\to64\to2\), again with two `tanh` hidden layers and a
linear output, for 4,610 parameters.  The complete-trajectory split contains
4,096/1,024/1,024 train/validation/test orbits.  Each orbit contributes 96
finite-time transitions, giving 393,216/98,304/98,304 rows.  The optimizer,
batch size, epoch ceiling, patience, and three seeds match the local protocol.
Seed 202 is the selected checkpoint, chosen only by validation
orbit-averaged standardized target MSE.

## Output-specialization control

The shared-head control is the selected \(4\to64\to64\to2\) model.  The
late-branching alternative is

$$
4\to64\to(32,32)\to(1,1),
$$

with 4,546 parameters.  The comparison is parameter-count-controlled, not
exactly parameter matched.  It was validation-only and does not replace the
selected shared-head model.

## Historical names retained in code

| Internal name | Public scientific meaning | Policy |
|---|---|---|
| `phase_b` | complete exact orbit banks | Retained in paths and schemas for reproducibility. |
| `phase_c` | finite-time transition dataset | Retained in paths and schemas for reproducibility. |
| `hybrid` | time-rescaled finite-time target representation | Retained for old manifests/checkpoints; public aliases are available in `finite_time_hybrid.py`. |
| `microcore` | targeted local sampling near the difficult incoming, near-zero-\(|\xi|\) region | Define once in technical documentation; omit from figure titles and narrative headings. |
| `Model A`, `A64`, `C32x32` | development-stage local architectures | Use only when reconstructing the local-model progression. |
