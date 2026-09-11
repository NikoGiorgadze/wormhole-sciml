# How this Python project works

This guide is written for someone who understands the physics and mathematics,
but is still learning how a Python research project is assembled. The goal is
to show both **what happens** during a calculation and **where it happens**.

You do not need to understand every Python detail before running the code. A
useful first pass is:

1. Read the architecture and calculation-flow sections.
2. Open `scripts/make_figures.py` and follow one figure function.
3. Use the later sections when unfamiliar syntax appears.

## 1. The main architecture

The project has four simple layers:

```text
User-facing scripts
    scripts/make_figures.py
    scripts/validate_baseline.py
            │
            │ import and call
            ▼
Scientific Python package
    parameters.py  →  geometry.py  →  dynamics.py
                                      │
                                      ▼
                                  integrate.py
                                      │
                                      ▼
                                 observables.py
            │
            │ return arrays and solver results
            ▼
Outputs
    output/physics_figures/*.png
    output/validation.json

Independent checks
    tests/test_reference_physics.py
    tests/test_public_model_contracts.py
```

The arrows describe the main flow, not rigid restrictions. For example,
`observables.py` also imports the areal-radius function from `geometry.py`.

The most important distinction is between the **package** and the **scripts**:

- `src/wormhole_sciml/` is the reusable scientific package. It defines the
  model and performs calculations. It does not decide which publication-style
  figures to make.
- `scripts/` contains small programs that choose parameter values, request
  calculations, and write output files.

This separation is useful in research. You can change a line color or choose a
new initial state without touching the acceleration equation. Conversely, you
can correct an equation in `dynamics.py` and every script automatically uses
the corrected function.

## 2. What happens during one trajectory calculation

Consider the through-wormhole calculation in `make_through_wormhole_dynamics`.
Its path through the project is:

```text
1. make_figures.py creates parameter objects
       WormholeParameters(m=2)
       SpiralParameters(omega=1, alpha=-1.25, theta=pi/6)

2. make_figures.py creates requested output times
       times = np.linspace(0, 200, 2001)

3. make_figures.py calls integrate_trajectory(...)
       initial state = (l₀, v₀) = (-30, 0.827)

4. integrate.py validates the initial state
       Is the state finite?
       Is the time interval increasing?
       Is v_tot² < 1?

5. integrate.py calls scipy.integrate.solve_ivp(...)

6. SciPy repeatedly calls dynamics.radial_system(...)
       input state:  [l, v]
       output:       [dl/dt, dv/dt] = [v, a(l,v)]

7. radial_system calls radial_acceleration(...)
       dynamics.py evaluates the physical equation of motion

8. SciPy chooses time steps, estimates its error, and returns a solution
       solution.t     contains times
       solution.y[0]  contains l(t)
       solution.y[1]  contains v(t)

9. make_figures.py calculates additional quantities
       total_speed_squared(l(t), v(t), ...)

10. Matplotlib plots the arrays and saves a PNG
```

SciPy may evaluate the equation at many internal times that are not in
`times`. The array passed as `t_eval=times` specifies where you want the final
solution reported; it does not force the adaptive solver to use those as its
internal steps.

## 3. What belongs in each file

### `parameters.py`: the numbers defining a physical case

This file defines two dataclasses:

```python
wormhole = WormholeParameters(throat_radius=1.0, m=2)
spiral = SpiralParameters(omega=1.0, alpha=-1.25, theta=np.pi / 6)
```

Think of each object as a small labelled parameter record. Instead of passing
an ambiguous list such as `[1.0, 2]`, the code can say `wormhole.m` or
`wormhole.throat_radius`.

The classes also reject impossible inputs when they are created. For example,
an odd `m` raises an error immediately. This is much easier to diagnose than a
solver failing later because it received an invalid geometry.

### `geometry.py`: functions determined by the metric

This file contains:

- `areal_radius`: calculates `R(l)`.
- `areal_radius_derivative`: calculates its derivative.
- `reduced_metric`: constructs the two-by-two metric after applying the spiral
  constraint.
- `reduced_metric_derivative`: differentiates that metric with respect to `l`.
- `metric_determinant`: evaluates the analytic determinant.

This module knows about the geometry, but it does not integrate trajectories.
That makes it possible to test metric identities independently of the ODE.

### `dynamics.py`: the physical motion and admissibility conditions

This is the central physics file. It contains three related groups:

1. Physical-domain functions such as `total_speed_squared`,
   `timelike_margin`, `is_admissible`, and `velocity_bounds`.
2. Conserved-quantity functions such as `lorentz_factor`,
   `conserved_energy`, and `velocity_from_energy`.
3. Evolution functions: `radial_acceleration` and `radial_system`.

The distinction between the last two is important:

```python
radial_acceleration(l, v, wormhole, spiral)
```

returns only the scalar acceleration `a(l, v)`. SciPy, however, needs the
derivative of the entire state vector. Therefore:

```python
radial_system(t, [l, v], wormhole, spiral)
```

returns:

```text
[v, a(l, v)]
```

This is the standard conversion of one second-order equation into two
first-order equations.

`radial_acceleration_from_metric` is intentionally separate from
`radial_acceleration`. It calculates the acceleration by a different algebraic
route. Agreement between the two is a scientific check, not duplicated code
needed for normal integration.

### `integrate.py`: the numerical-solver boundary

This file is the bridge between our functions and SciPy. It:

- checks the requested initial state;
- defines the event that detects the null boundary;
- passes `radial_system` to `solve_ivp`;
- supplies tolerances and solver options;
- checks that SciPy completed successfully;
- returns SciPy's solution object.

The equations are not copied into this file. The solver receives the equation
through the function `radial_system`.

### `observables.py`: quantities calculated after or alongside integration

This file contains transformations that are useful but are not part of the
two-component ODE:

- effective angular velocity;
- terminal radial velocity;
- azimuth in laboratory or co-rotating frames;
- Cartesian-like coordinates for three-dimensional plots.

For example, the solver only needs to determine `l(t)` and `v(t)`. The plotting
code later combines `l(t)` with the azimuth formula to obtain `x(t)`, `y(t)`,
and `z(t)`. Adding Cartesian coordinates to the ODE state would make the solver
larger without adding physical information.

### `__init__.py`: the public front door of the package

This file gathers the functions intended for ordinary use. Because of it, a
script can write:

```python
from wormhole_sciml import integrate_trajectory
```

instead of needing to know that the function lives in:

```python
from wormhole_sciml.integrate import integrate_trajectory
```

No physics is calculated merely because a name is listed in `__init__.py`. It
only makes imports shorter and defines the package's public vocabulary.

### `scripts/make_figures.py`: choose cases and present results

Each `make_...` function creates one figure. A figure function usually does
the following:

1. Selects parameters and initial conditions.
2. Creates a time or phase-space grid.
3. Calls package functions.
4. Plots the returned arrays.
5. Saves one PNG file.

The `main()` function calls all figure functions. This block at the end:

```python
if __name__ == "__main__":
    main()
```

means: call `main()` when the file is executed as a program. It prevents the
figures from being generated merely because another file imports something
from this script.

### `scripts/validate_baseline.py`: slower scientific checks

This script runs quantitative comparisons, such as agreement between two
acceleration formulas and sensitivity to tighter solver tolerances. It writes
the measured errors to an ignored file under `output/` so the numerical evidence is
saved rather than existing only as terminal output.

### `tests/test_reference_physics.py`: fast automatic checks

Tests are small calculations with a definite expected result. They answer
questions such as:

- Does the metric determinant equal its analytic formula?
- Does a terminal state have the expected acceleration?
- Is an inadmissible initial state rejected?

Tests should remain fast enough to run after small edits. More expensive
convergence studies stay in the validation script.

## 4. How imports join the files together

Inside the package, an import such as

```python
from .parameters import SpiralParameters, WormholeParameters
```

uses a leading dot. The dot means "from another module in this same package."
This lets `dynamics.py` use the parameter classes without copying them.

At the project level, the command prefix

```bash
PYTHONPATH=src
```

tells Python to include the `src/` directory when looking for importable
packages. Python then finds `src/wormhole_sciml/__init__.py` when it sees:

```python
import wormhole_sciml
```

An import loads function and class definitions. A function body runs only when
the function is called.

## 5. Parameter classes and dataclass syntax

The declaration

```python
@dataclass(frozen=True, slots=True)
class WormholeParameters:
```

contains several ideas:

- `class` creates a new kind of object.
- `@dataclass(...)` asks Python to generate the routine that stores the named
  fields.
- `frozen=True` prevents an object's values from being changed accidentally
  during a calculation.
- `slots=True` prevents accidental new attributes caused by spelling mistakes
  and keeps the object small.

The field declarations provide defaults:

```python
throat_radius: float = 1.0
m: int = 2
```

Therefore these are both valid:

```python
WormholeParameters()       # uses b0=1.0 and m=2
WormholeParameters(m=10)   # changes m but keeps b0=1.0
```

`__post_init__` runs immediately after the generated initialization routine.
It is used to validate the newly stored parameters.

## 6. Functions, arguments, and returned values

A function definition has a name, inputs, and a body:

```python
def terminal_radial_velocity(spiral: SpiralParameters) -> float:
    return -spiral.omega / spiral.alpha
```

Here:

- `spiral` is the input variable.
- `SpiralParameters` is a type hint describing the expected input.
- `-> float` is a type hint describing the result.
- `return` sends the calculated value back to the caller.

Type hints help readers and editors; ordinary Python does not enforce every
hint while the program runs.

A docstring is the triple-quoted text immediately below a function or class
definition. In an interactive session:

```python
help(terminal_radial_velocity)
```

shows that explanation.

## 7. Arrays, shapes, and vectorized calculations

NumPy arrays allow the same formula to operate on one number or many numbers.

```python
l = np.linspace(-20.0, 20.0, 1001)
radius = areal_radius(l, wormhole)
```

`l` has shape `(1001,)`: it is a one-dimensional collection of 1001 numbers.
`radius` has the same shape because `areal_radius` evaluates every element.

This is called vectorization. It avoids a Python loop for a formula that NumPy
can apply to an entire array.

The geometry functions begin with code like:

```python
l = np.asarray(proper_radius, dtype=np.float64)
```

This converts a Python number, list, or existing array into NumPy's
double-precision format. Consequently the public function accepts either:

```python
areal_radius(2.0, wormhole)
areal_radius([0.0, 1.0, 2.0], wormhole)
```

### Broadcasting in a phase-space grid

For a vector field, every selected `l` must be combined with every selected
`v`:

```python
l_values = np.linspace(0.0, 10.0, 24)
v_values = np.linspace(0.0, 0.9, 23)
l_grid, v_grid = np.meshgrid(l_values, v_values)
a_grid = radial_acceleration(l_grid, v_grid, wormhole, spiral)
```

Both grids have shape `(23, 24)`. At position `[i, j]`, `v_grid` contains the
`i`th velocity and `l_grid` contains the `j`th radial coordinate. The returned
acceleration grid has one result for every phase-space point.

### Reading the SciPy solution array

For the two-component state `[l, v]`, `solution.y` has shape `(2, N)`:

```text
solution.y
    row 0  →  l at every reported time
    row 1  →  v at every reported time
```

Python indices start at zero, so `solution.y[0]` means the first row.

## 8. How SciPy receives our equation

`solve_ivp` expects a callable with the general form:

```python
derivative = function(time, state, extra_arguments...)
```

Our callable is `radial_system`. In `integrate.py`, it is passed like this:

```python
solve_ivp(
    radial_system,
    t_span,
    state,
    args=(wormhole, spiral),
)
```

Notice that `radial_system` has no parentheses. This passes the function itself
to SciPy. Writing `radial_system(...)` would run it immediately instead.

SciPy later makes calls equivalent to:

```python
radial_system(current_time, current_state, wormhole, spiral)
```

It repeats this while choosing steps and estimating numerical error.

### The stopping event

`null_boundary` is defined inside `integrate_trajectory` because nothing else
needs it. It returns the timelike margin:

```text
positive value  →  massive-particle region
zero            →  null boundary
negative value  →  unphysical continuation
```

These attributes give instructions to SciPy:

```python
null_boundary.terminal = True
null_boundary.direction = -1.0
```

They mean "stop at the detected zero" and "look for a crossing from positive
to negative."

## 9. Python syntax used often in the figure script

### Tuples and unpacking

An initial state is stored as a two-value tuple:

```python
initial_state = (-30.0, 0.827)
```

This assignment separates a two-value state:

```python
proper_radius, velocity = state
```

It is equivalent to assigning `state[0]` and `state[1]` separately.

### Keyword arguments

This call uses names for clarity:

```python
integrate_trajectory(
    initial_state=(-30.0, 0.827),
    t_span=(0.0, 200.0),
    wormhole=wormhole,
    spiral=spiral,
    t_eval=times,
)
```

The standalone `*` in the function definition means that the later options,
such as `rtol`, must be given by name. This prevents a long list of numerical
arguments whose meanings are difficult to remember.

### Dictionaries and double-star unpacking

The figure script stores shared solver settings in a dictionary:

```python
SOLVER = {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12}
```

The call:

```python
integrate_trajectory(..., **SOLVER)
```

is a compact form of:

```python
integrate_trajectory(
    ...,
    method="DOP853",
    rtol=1e-10,
    atol=1e-12,
)
```

The two stars unpack dictionary keys as named arguments.

### Loops with `zip`

This loop advances through two sequences together:

```python
for color, m in zip(colors, M_VALUES):
    wormhole = WormholeParameters(m=m)
```

On each iteration, `color` and `m` receive corresponding elements. This is
cleaner than manually managing an integer loop index.

### `None`

`None` means "no value was supplied." For example, `t_eval=None` tells SciPy
that it may return values at its own accepted time steps.

### Exceptions

Code such as:

```python
raise ValueError("initial_state is not timelike")
```

stops the calculation with a precise explanation. It prevents invalid input
from silently producing misleading output.

## 10. From a solution to a three-dimensional figure

The solver returns only `l(t)` and `v(t)`. A three-dimensional plot is built in
three stages:

```text
l(t), t
   │
   ▼
azimuth(...)
   Φ(t) = αl(t) + ωt + phase
   │
   ▼
coordinate_visualization(...)
   x(t), y(t), z(t)
   │
   ▼
Matplotlib axis.plot(x, y, z)
```

For a multi-particle plot, the code reuses the same radial solution and changes
only `phase`. There is no need to integrate twelve identical radial equations.
This is both simpler and faster.

The frame option changes the azimuth calculation:

- `frame="laboratory"` includes the `omega * time` rotation.
- `frame="co_rotating"` omits that rotation.

## 11. Paths and generated files

The scripts use `pathlib.Path`:

```python
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)
figure.savefig(output_dir / "vector_fields.png")
```

For `Path` objects, `/` joins path components; it does not mean division.
`parents=True` permits creation of missing parent directories, and
`exist_ok=True` means an existing output directory is acceptable.

Generated results belong in `reports/` or `output/`, not in the source package.
The source remains readable even after many calculations have been run.

## 12. How to make common changes

### Change a parameter or initial condition for a figure

Edit the relevant `make_...` function in `scripts/make_figures.py`. Change the
parameter object or the `initial_state` passed to `integrate_trajectory`. You do
not need to edit the package equations.

### Add a new plot of an existing quantity

Add a new function to `scripts/make_figures.py`. Inside it:

1. Create parameters.
2. Integrate if necessary.
3. Calculate the desired arrays.
4. Plot and save them.
5. Add one call to the new function inside `main()`.

### Add a derived physical quantity

If the quantity can be calculated from an existing trajectory and does not
change the ODE, add a pure function to `observables.py`. Export it through
`__init__.py` if scripts should import it from `wormhole_sciml`.

### Change the geometry

Geometry formulas belong in `geometry.py`. A genuinely different metric may
also require changes in `dynamics.py`; after such a change, run both tests and
the numerical validation.

### Change the equation being integrated

Edit `radial_acceleration` in `dynamics.py`, then check whether
`radial_acceleration_from_metric` should change independently. Run the tests
before regenerating figures. A plotting script is not the right place to alter
the equation of motion.

### Add another state variable

This is a larger change. You would need to:

1. Add the variable to the state accepted by `radial_system`.
2. Return its derivative in the same position.
3. Update initial-state validation in `integrate.py`.
4. Update every place that interprets rows of `solution.y`.
5. Add tests for the new equation.

This illustrates why the order of the state vector must be documented.

## 13. A practical learning and debugging routine

Run commands from the project root.

First run the fast tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Then run the deeper numerical checks:

```bash
PYTHONPATH=src python3 scripts/validate_baseline.py
```

Finally regenerate figures:

```bash
PYTHONPATH=src python3 scripts/make_figures.py
```

When experimenting, change one thing at a time. If something fails:

1. Read the last line of the error first; it usually states the exception.
2. Read upward to find the first project filename in the traceback.
3. Print the type and shape of suspicious values:

   ```python
   print(type(solution.y), solution.y.shape)
   ```

4. Check the physical domain before blaming the solver.
5. Re-run the tests after the fix.

The central mental model is this: parameter objects describe a case, pure
functions express the geometry and physics, `integrate.py` hands the evolution
function to SciPy, observables transform the returned trajectory, and scripts
turn the arrays into saved scientific results.

The project extends the same separation to dataset generation, local models,
and finite-time models. See `model_definitions.md` for the public model
interfaces and `reproducibility.md` for the complete experiment order.
