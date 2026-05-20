# Self-Consistent Interneuron Solving

## The problem

When computing the PYR nullcline $F(r_\text{PYR})$, we sweep $r_\text{PYR}$ over a range of values and ask:
*"If PYR fires at this rate, where do the other populations land at steady state?"*

VIP is simple — its input depends only on $r_\text{PYR}$ and external drive, so its steady state is a direct one-shot computation:

$$r_\text{VIP} = \Phi_\text{VIP}(w_{EV} \cdot r_\text{PYR} + I_\text{ext,VIP})$$

SOM and PV are harder. They inhibit each other, so you cannot solve one without knowing the other:

- SOM receives inhibition from VIP
- PV receives inhibition from both SOM and PV itself (self-inhibition)

This **mutual dependency** means there is no explicit formula — you must find the pair $(r_\text{SOM}, r_\text{PV})$ that satisfies both steady-state equations at once.

---

## Formulating it as a root-finding problem

At steady state, each population's firing rate equals the output of its transfer function:

$$r_\text{SOM} = \Phi_\text{SOM}(I_\text{SOM}(r_\text{PYR},\, r_\text{VIP},\, r_\text{SOM}))$$
$$r_\text{PV}  = \Phi_\text{PV} (I_\text{PV} (r_\text{PYR},\, r_\text{VIP},\, r_\text{SOM},\, r_\text{PV}))$$

Rearranged as **residuals** — quantities that must equal zero at the solution:

$$g_1(r_\text{SOM}, r_\text{PV}) = \Phi_\text{SOM}(\ldots) - r_\text{SOM} = 0$$
$$g_2(r_\text{SOM}, r_\text{PV}) = \Phi_\text{PV}(\ldots)  - r_\text{PV}  = 0$$

Finding $(r_\text{SOM}, r_\text{PV})$ such that both residuals vanish simultaneously is a **2D root-finding problem**.

---

## Why not solve analytically?

$\Phi$ (the Wong-Wang transfer function) is nonlinear — there is no closed-form inverse. We must solve numerically.

---

## Newton's method (what `fsolve` does)

`scipy.optimize.fsolve` implements Newton's method for systems of equations. The idea:

1. **Start** from an initial guess, e.g. $(r_\text{SOM}, r_\text{PV}) = (0, 0)$
2. **Evaluate** the residuals $g_1$ and $g_2$ at the current guess — how far off are we?
3. **Linearize** — compute how $g_1$ and $g_2$ change as you nudge each variable (the 2×2 Jacobian of $g$)
4. **Step** in the direction that would bring both residuals to zero if $g$ were linear
5. **Repeat** until $|g_1|$ and $|g_2|$ are below a tolerance

Because $\Phi$ is smooth, this converges in a handful of iterations.

### Small example

Suppose at the current guess $(0, 0)$ you find $g_1 = +3$ (SOM is too low) and $g_2 = -1$ (PV is too high). The solver computes the local slopes:

$$\frac{\partial g_1}{\partial r_\text{SOM}},\quad \frac{\partial g_1}{\partial r_\text{PV}},\quad \frac{\partial g_2}{\partial r_\text{SOM}},\quad \frac{\partial g_2}{\partial r_\text{PV}}$$

and takes a step that reduces both residuals jointly. It then re-evaluates, steps again, and so on.

---

## Two initial guesses

Newton's method can get stuck or converge to a spurious solution if the starting point is poor. The code therefore tries **two starting points**: $(0, 0)$ (both silent) and $(30, 30)$ (both moderately active). It keeps whichever converged solution has the smaller total residual — a cheap way to guard against local failures.

```python
for x0 in [(0.0, 0.0), (30.0, 30.0)]:
    sol = fsolve(residuals, x0, full_output=True)
    ...
_, (r_som, r_pv) = min(results, key=lambda t: t[0])
```

Source: [`circuit_model/bistable_loss.py:121-128`](../circuit_model/bistable_loss.py#L121-L128)

---

## What this gives us

At every point $r_\text{PYR}$ in the sweep, the solver returns the **self-consistent** interneuron rates — the values that SOM and PV would actually settle to if PYR were held fixed at that rate. This means the nullcline $F(r_\text{PYR})$ and its zeros (the fixed points) are true 4-population steady states, not approximations.

The full picture:

```
for each r_PYR in sweep:
    1. r_VIP  ← direct formula
    2. (r_SOM, r_PV) ← fsolve (Newton's method on 2-equation residual)
    3. I_net(r_PYR) ← combine all four rates
    4. F(r_PYR) = Φ_PYR(I_net) - r_PYR
```

Zeros of $F$ are fixed points of the full 4-population circuit.
