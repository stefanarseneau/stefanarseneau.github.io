---
layout: note
title:  "Derive Every Dispersion Relation!"
---

Notes based on [Lecture Notes on Stellar Oscillations](https://users-phys.au.dk/jcd/oscilnotes/Lecture_Notes_on_Stellar_Oscillations.pdf) by Jorgen Christensen-Dalsgaard.

<p>
    <h2>Table of Contents ---</h2>
<list>
    <li><a href="#cartesian-acoustic">01 - The Acoustic Dispersion Relation</a></li>
    <li><a href="#cartesian-gravity">02 - Gravity Wave Dispersion Relation</a></li>
</list>
<hr>
</p>


<h2 id="cartesian-acoustic">01 - The Acoustic Dispersion Relation</h2>

I'm going to start off by deriving the dispersion relations for p-modes, g-modes, and surface g-modes in Cartesian coordinates, such as one might find in the ocean (hence "waves for oceanographers"). I'm starting here because it's more intuitive to me to think about these things in Cartesian coordinates than in spherical coordinates, where we have to introduce spherical harmonics. The equations that govern fluid flow in these circumstances are going to be the continuity equation, which dictates the condition that mass is conserved; the Navier-Stokes (or momentum) equation, which governs conservation of momentum; and an equation of state which I'll just take to be polytropic. In the absence of terms like viscosity, these can be written as:

$$\frac{\partial \rho}{\partial t} + \nabla\cdot(\rho\vec{v}) = 0$$

$$\rho \frac{\partial \vec{v}}{\partial t} + \rho(\vec{v}\cdot\nabla)\vec{v} = -\nabla P + \rho\vec{g}$$

$$p = \kappa \rho^{\gamma}$$

These equations are nonlinear and complex, so the general approach is going to be to linearize them around an equilibrium state and solve for perturbations. In particular, we'll want to write pressure, density, and velocity in terms of an equillibrium component and a perturbation component:

$$P = P_0 + P_1$$

$$\rho = \rho_0 + \rho_1$$

$$\vec{v} = \vec{v}_0 + \vec{v}_1 = \vec{v}_1$$

where we've taken $$\vec{v}_0 = 0$$ since in an equilibrium steady state there is no motion. In the equilibrium case then, we zero out all the time derivatives for all quantities and reduce to:

$$ \nabla\cdot(\rho\vec{v_0}) = 0 \implies \nabla\cdot\vec{v}_0 = 0$$

$$\nabla P_0 = \rho_0\vec{g}$$

The former is clearly true since $$\vec{v_0} = 0$$, and the latter is immediately recognizable as the condition for hydrostatic equilibrium. 

Now, we'll want to substitute the equilibrium and perturbation quantities into the momentum equation. We'll drop any second-order terms, which in this case means anywhere we're multiplying a perturbation term by another perturbation term (since that's going to result in something negligible).

$$
\begin{align}
    (\rho_0 + \rho_1) \frac{D \vec{v}_1}{D t} &= -\nabla P_0 - \nabla P_1 + (\rho_0 + \rho_1)(\vec{g}_0 + \vec{g}_1) \\
    \implies \rho_0 \frac{D\vec{v}_1}{Dt} &= -\nabla P_1 + \rho_0 \vec{g}_1 + \rho_1\vec{g}_0 \\
    \implies \rho_0 \frac{D\vec{v}_1}{Dt} &= -\nabla P_1 + \rho_1\vec{g}_0
\end{align}
$$

where we've just made the assumption that the gravitational potential will remain constant. It turns out that we can neglect the gravity term here as well. This is because we have required that the spatial derivatives of all equilibrium quantities be zero, and the condition of hydrostatic equilibrium then tells us that

$$\nabla P_0 = 0 = \rho_0\vec{g}_0$$

and since clearly $$\rho_0$$ must be nonzero, it must be the case that the equilibrium gravity term is negligible. Then,

$$\frac{D\vec{v}_1}{Dt} = -\frac{1}{\rho_0}\nabla P_1$$

Now we can take the divergence of both sides of the equation and use the fact that the velocity is the first derivative of some position perturbation to show that:

$$\nabla\cdot\frac{D\vec{v}_1}{Dt} = \frac{D^2}{Dt^2}(\nabla\cdot \vec{\delta r}) =  -\frac{1}{\rho_0}\nabla^2 P_1$$

If we express the continuity equation in terms of a position perturbation rather than a velocity perturbation, we can find something more useful in all of this. We've already shown that in the equilibrium case, it just reduces to the zero velocity condition. In the perturbation case though, we can get:

$$
\begin{align}
    \frac{d\rho}{dt} + \nabla\cdot(\rho_0\vec{v}) &= 0 \implies \frac{\vec{\delta \rho}}{\delta t} + \nabla\cdot(\rho_0\frac{\vec{\delta r}}{\delta t}) = 0 \\
    \implies \vec{\delta \rho} + \rho_0\nabla\cdot(\vec{\delta r}) &= 0 \\
    \implies \nabla\cdot\vec{\delta r} &= -\frac{\delta\rho}{\rho_0} = -\frac{\rho_1}{\rho_0}
\end{align}
$$

which intuitively means that the divergence of position is equal to minus the strength of the density perturbation. Let's now put this into the momentum equation:

$$ \frac{D^2}{Dt^2}(\nabla\cdot \vec{\delta r}) =  -\frac{1}{\rho_0}\nabla^2 P_1 \implies  \frac{D^2\rho_1}{Dt^2} = \nabla^2 P_1$$

Finally, we can apply the equation of state to put everything in terms of pressure. First, we can see that the specified equaiton of state $$p = \kappa \rho^\gamma$$ also implies that 

$$\frac{\delta P}{P} = \gamma \frac{\delta \rho}{\rho} \implies \rho_1 = \frac{\rho_0}{\gamma P_0} P_1$$

and so substituting in, we get the acoustic wave equation

$$ \frac{D^2P_1}{Dt^2} = \frac{\gamma P_0}{\rho_0} \nabla^2 P_1 = c_s^2 \nabla^2 P_1$$

This is a wave equation, so we can assume a solution of the form $$P_1 = a\exp(i(\vec{k}\cdot\vec{r} - \omega t))$$, which gives us the dispersion relation by substituting in:

$$ \omega^2 = c_s^2 |\vec{k}|^2 $$

<h2 id="cartesian-gravity">02 - Internal Gravity Wave Dispersion Relation</h2>

The method we used for the acoustic wave equation works quite well, but it's also complicated. When we think about internal gravity waves, we can really just assume that the perturbation quantities will vary like $$\exp[i(\vec{k}\cdot\vec{r} - \omega t)]$$. We'll also assume that equilibrium quantities vary slowly and that perturbations to the gravity field are negligible just like we did before. Taking the $$\hat{r}$$ coordinate to be vertically oriented, we can imagine that the only components of the pressure and density gradients are in that direction since the only external force is gravity:

$$\nabla p_0 = \frac{dp_0}{dr}\hat{r}$$

$$\nabla \rho_0 = \frac{d\rho_0}{dr}\hat{r}$$

So this is a problem that has a preferred direction. It makes sense then to split the quantities that we're most interested in solving for (displacement $$\vec{\delta r}$$ and wavenumber $$\vec{k}$$) into two components: the radial and horizontal:

$$\vec{\delta r} = \xi_r \hat{r} + \vec{\xi}_H$$

$$\vec{k} = k_r\hat{r} + \vec{k}_H$$

and likewise, it's going to make sense to split the momentum equation up into these two directions:

$$
\begin{align}
    \rho_0\frac{d^2 \vec{\delta r}}{dt^2} &= -\nabla p_1 + \rho_0\vec{g}_1 + \rho_1\vec{g}_0 \\
    \implies \rho_0\frac{d^2\xi_r}{dt^2} &= \frac{dp_1}{dr} - \rho_1g_0 \\
    \implies \rho_0\frac{d^2\xi_H}{dt^2} &= \nabla_H p_1
\end{align}
$$

which, when we substitute in the assumed form of the perturbation quantities, gives us:

$$-\rho_0\omega^2\xi_r = -ik_rp_1 - \rho_1g_0$$

$$ -\rho_0\omega^2\vec{\xi_H} = -i\vec{k}_H p_1$$

Now, since we've assumed a form for the perturbation terms, we can actually get more information out of the continuity equation. We'll do this by looking at the continuity equation as it relates to perturbation quantities:

$$\frac{d\rho}{dt} + \nabla\cdot(\rho_0\vec{v}) \approx \frac{\delta\rho}{\delta t} + \nabla\cdot(\rho_0\frac{\vec{\delta r}}{\delta t}) = 0$$

$$\implies \delta \rho + \nabla\cdot(\rho_0\vec{\delta r}) = 0$$

so that substituting in $$\delta \rho = \rho_1$$ and the above expression for displacement, we get:

$$\rho' + \rho_0 ik_r\xi_r + \rho_0 i \vec{k}_H\cdot \vec{\xi}_H = 0$$

We can use the expression $$ -\rho_0\omega^2\vec{\xi_H} = -i\vec{k}_H p_1$$ to get a relation between the pressure and density perturbations:

$$
\begin{align}
    0 &= \rho' + \rho_0 ik_r\xi_r + \rho_0 i \vec{k}_H\cdot \vec{\xi}_H &= \rho' + \rho_0 ik_r\xi_r + \rho_0 i \vec{k}_H\cdot \left(\frac{ip_1}{\rho_0\omega^2}\vec{k}_H\right) \\
    &= \rho_1 + \rho_0i k_r\xi_r - |\vec{k}_h|^2\frac{p_1}{\omega^2} \\
    \implies p_1 &= \frac{\omega^2}{|\vec{k}_h|^2}(\rho_1 + \rho_0 ik_r \xi_r)
\end{align}
$$

This gives us a term which we can substitute into the radial form of the Navier-Stokes equation: $$-\rho_0\omega^2\xi_r = -ik_rp_1 - \rho_1g_0$$, to get

$$
\begin{align}
    -\rho_0\omega^2\xi_r &= -ik_r\left[\frac{\omega^2}{|\vec{k}_h|^2}(\rho_1 + \rho_0 ik_r \xi_r)\right] - \rho_1g_0 \\
    &= -i\frac{k_r}{k_H^2}\omega^2\rho_1 + \omega^2\rho_0\frac{k_r^2}{k_h^2}\xi_r - \rho_1g_0 
\end{align}
$$