---
layout: note
title:  "Duffing Oscillators and Strange Attractors"
category: stochastic 
---

<ins>*Subtitle:*</ins> I make some pretty pictures with Runge-Kutta integration.



The Duffing oscillator is a differential equation which governs the motion of a damped particle in a double-welled potential which is being driven by a sinusoidal external force at some frequency $$\omega_0$$. The equation of motion of the system is

$$m\ddot{x} = -\gamma\dot{x} + 2ax - 4bx^3 + F_0\cos\omega_0 t$$

where $$\gamma$$ is the damping coefficient. I'm going to code up a fourth-order Runge-Kutta integrator for this equation in python, and use it to visualize the chaotic phase space behavior of the system. The numerical parameters and libraries I'll use are:

```python
import matplotlib.pyplot as plt
import numpy as np

m = 1.00 ; a = 0.5 ; b = 0.25
F0 = 2.0 ; om = 2.4 ; gam = 0.1
```

Of course this means that the acceleration of the particle is expressed in code as 

```python
def acceleration(x, xp, t):
    return (-gam*xp + 2*a*x - 4*b*x**3 + F0*np.cos(om*t)) / m
```

All that remains is to integrate the equation. I do this with a quick RK4 solver which integrates the two equations

$$\dot{y} = -\gamma y + 2ax - 4bx^3 + F_0\cos\omega_0t$$

$$\dot{x} = y$$

which we do by computing the acceleration function, which we'll call $$f(x, y, t)$$, at special points. At each step, we do:

$$k_{1,x}^i = dt \dot{x}^{i-1} \qquad \text{  } \qquad k_{1,v}^i = dt f(x^{i-1}, y^{i-1}, t)$$

$$k_{2,x}^i = dt (\dot{x}^{i-1} + 0.5k_{1,v}^i) \qquad \text{  } \qquad k_{2,v}^i = dt f(x^{i-1} + 0.5 k_{1,x}^i, y^{i-1} + 0.5 k_{1,v}^i, t)$$

$$k_{3,x}^i = dt \dot{x}^{i-1} (\dot{x}^{i-1} + 0.5k_{2,v}^i) \qquad \text{  } \qquad k_{2,v}^i = dt f(x^{i-1} + 0.5 k_{2,x}^i, y^{i-1} + 0.5 k_{2,v}^i, t)$$

$$k_{4,x}^i = dt \dot{x}^{i-1} (\dot{x}^{i-1} + k_{3,v}^i) \qquad \text{  } \qquad k_{2,v}^i = dt f(x^{i-1} + k_{3,x}^i, y^{i-1} + k_{3,v}^i, t)$$

and using this we step forward in time by

$$x^i = x^{i-1} + \frac{1}{6}\left(k_{1,x}^i + 2k_{2,x}^i + 2k_{3,x}^i + k_{4,x}^i\right)$$

$$y^i = y^{i-1} + \frac{1}{6}\left(k_{1,v}^i + 2k_{2,v}^i + 2k_{3,v}^i + k_{4,v}^i\right)$$

In python, this looks like

```python
def rungekutta(x0, xp0, dt, Np):
    Tpoinc = 2*np.pi / om
    xpoinc, xppoinc = [], []

    tt = np.arange(0, Np*Tpoinc, dt)
    xx = np.zeros_like(tt)
    xp = np.zeros_like(tt)
    xx[0] = x0 ; xp[0] = xp0
    
    for ii in range(1, tt.shape[0]):
        k1v = dt*acceleration(xx[ii-1], xp[ii-1], tt[ii-1])
        k1x = dt*xp[ii-1]

        k2v = dt*acceleration(xx[ii-1] + 0.5*k1x, xp[ii-1] + 0.5*k1v, tt[ii-1] + 0.5*dt)
        k2x = dt*(xp[ii-1] + 0.5*k1v)

        k3v = dt*acceleration(xx[ii-1] + 0.5*k2x, xp[ii-1] + 0.5*k2v, tt[ii-1] + 0.5*dt)
        k3x = dt*(xp[ii-1] + 0.5*k2v)

        k4v = dt*acceleration(xx[ii-1] + k3x, xp[ii-1] + k3v, tt[ii-1] + dt)
        k4x = dt*(xp[ii-1] + k3v)

        xx[ii] = xx[ii-1] + (k1x + 2*k2x + 2*k3x + k4x) / 6
        xp[ii] = xp[ii-1] + (k1v + 2*k2v + 2*k3v + k4v) / 6

        if tt[ii] % Tpoinc < dt:
            xpoinc.append(xx[ii])
            xppoinc.append(xp[ii]) 

    return tt, xx, xp, (xpoinc, xppoinc)
```

The final if statement in that loop picks out points in the integration which are at a set phase of the driving frequency. We can visualize the chaotic behavior of the system by plotting these points in phase space. This section of the oscillator is called a Poincare section, and it characterizes the dynamical response of the system. Running the simulation for 5000 periods of the driving frequency with a sufficiently fine time step, we get a sufficiently dense sampling of phase space to make out the following shape:

<br>

```python
tt1, xx1, xp1, poincare = rungekutta(0.5, 0, dt=0.01, Np=5000)

plt.figure(figsize=(12, 8))
plt.scatter(poincare[0], poincare[1], s=5, c = 'k')
plt.xlabel("Position")
plt.ylabel("Velocity")
```

<br>

<p><center>
  <img src="/images/duffing/duffing_attractor.png"  width="50%"/>
</center></p>

<br>

This is a strange attractor; it is a fractal with dimension of 1.67. The dimension of a fractal is defined by measure theory (specifically using a Hausdorff measure), but intuitively we can imagine trying to cover this curve in $n$-dimensional segments. A perfect line could be covered by one-dimensional segments, and a circle could be covered by a large number of boxes but will not be covered by one-dimensional segments (because it is fundamentally a two-dimensional shape). Looking at this curve, we can see that it appears to have some sort of linear features but is clearly not a perfect line. It is in between one and two dimensions. 

We could write a code to calculate an approximate Hausdorff dimension, and maybe I will in the future.