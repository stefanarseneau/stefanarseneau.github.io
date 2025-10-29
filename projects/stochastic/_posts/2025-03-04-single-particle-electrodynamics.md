---
layout: note
title:  "Single Particle Electrodynamics"
---

<p>
    <h2>Table of Contents ---</h2>
<list>
    <li><a href="#drift">01 - Derivation of Curvature Drift</a></li>
    <li><a href="#polar">02 - Derivation of Curvature and Polarization Drift</a></li>
    <li><a href="#mirror">03 - Magnetic Mirrors</a></li>
    <li><a href="#pulse">04 - Waves</a></li>
</list>
<hr>
</p>

<h2 id="drift">01 - Derivation of Curvature Drift</h2>

Single particle motion in an electromagnetic field is governed by the Lorentz equation:

$$m\frac{d\vec{v}}{dt} = q\left[\vec{E} + \vec{v}\times\vec{B}\right]$$

This is simple enough in a uniform magnetic field, but becomes more complicated in a field where we take into account the particle's gyration in the magnetic field.

$$m\frac{d\vec{v}}{dt} = q\left[\vec{E}(\vec{x}_\text{gc}) + (\vec{r}_\text{L}\cdot\nabla)\vec{E}(\vec{x}_\text{gc}) + \vec{v}\times\left(\vec{B}(\vec{x}_\text{gc}) + (\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\right)\right]$$

To make analyzing this system easier, we're going to want to try and average over the L'Armor orbit of the gyrating particle. To do that, we can split its velocity up into two vectors: the velocity of the particle guiding center and that of the L'Armor gyration.

$$\vec{v} = \vec{v}_\text{gc} + \vec{v}_L$$

$$m\frac{d(\vec{v}_\text{gc} + \vec{v}_L)}{dt} = q\left[\vec{E}(\vec{x}_\text{gc}) + (\vec{r}_\text{L}\cdot\nabla)\vec{E}(\vec{x}_\text{gc}) + (\vec{v}_\text{gc} + \vec{v}_L)\times\left(\vec{B}(\vec{x}_\text{gc}) + (\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\right)\right]$$

Now, if we were to average over the L'Armor gyration, all the terms that are linear in $$\vec{v}_L$$ and $$\vec{r}_L$$ will go to zero since those are cyclic. That leaves only those terms which are either not dependent on L'Armor terms or are dependent to an even power.

$$m\frac{d\vec{v}_\text{gc}}{dt} = q\left[\vec{E}(\vec{x}_\text{gc})+ \vec{v}_\text{gc}\times\vec{B}(\vec{x}_\text{gc}) + \langle\vec{v}_L\times(\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\rangle\right]$$

This gives us the information we need to evaluate the gradient drift. To do that, we'll calculate the averaging of the interaction between the particle's L'Armor motion and the perturbations of the magnetic field. We can write the position of the particle in the frame of the guiding center as $$\vec{r}_L = r_L\sin\omega_L t\hat{x} + r_L\cos\omega_L t\hat{y}$$ making the L'Armor velocity $$\vec{v}_L = r_L\omega_L\cos\omega_L t\hat{x} - r_L\omega_L\sin\omega_L t\hat{y}$$ making the gradient force term:

$$\vec{F}_\nabla =  \langle q(r_L\omega_L\cos\omega_L t\hat{x} - r_L\omega_L\sin\omega_L t\hat{y})\times((r_L\sin\omega_L t\hat{x} + r_L\cos\omega_L t\hat{y})\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\rangle$$

$$\vec{F}_\nabla  =  qr_L^2\omega_L\left\langle \cos\omega_L t\sin\omega_L t (\hat{x}\times(\hat{x}\cdot\nabla)\vec{B}) + \cos^2\omega_L t (\hat{x}\times(\hat{y}\cdot\nabla)\vec{B}) - \cos\omega_L t\sin\omega_L t (\hat{y}\times(\hat{x}\cdot\nabla)\vec{B}) - \sin^2\omega_L t (\hat{y}\times(\hat{x}\cdot\nabla)\vec{B})  \right\rangle$$

$$\vec{F}_\nabla  =  qr_L^2\omega_L\left[\frac{1}{2} (\hat{x}\times(\hat{y}\cdot\nabla)\vec{B}) - \frac{1}{2} (\hat{y}\times(\hat{x}\cdot\nabla)\vec{B})  \right]$$

where the linear terms are killed by averaging over the orbit. Now, we can take the magnetic field to be of the form $$\vec{B} = B_x\hat{x} + B_y\hat{y} + B_z\hat{z}$$ such that

$$\vec{F}_\nabla  =  qr_L^2\omega_L\left[\frac{1}{2} \left(\hat{x}\times\left(\frac{dB_x}{dy}\hat{x} + \frac{dB_y}{dy}\hat{y} + \frac{dB_z}{dy}\hat{z}\right)\right) - \frac{1}{2}  \left(\hat{y}\times\left(\frac{dB_x}{dx}\hat{x} + \frac{dB_y}{dx}\hat{y} + \frac{dB_z}{dx}\hat{z}\right)\right)   \right]$$

$$\vec{F}_\nabla  =  \frac{qr_L^2\omega_L}{2} \left[\left(\frac{dB_y}{dy}\hat{z} - \frac{dB_z}{dy}\hat{y}\right) - \left(-\frac{dB_x}{dx}\hat{z} + \frac{dB_z}{dx}\hat{x}\right)\right]$$

$$\vec{F}_\nabla  =  \frac{qr_L^2\omega_L}{2} \left[\left(\frac{dB_y}{dy} + \frac{dB_x}{dx}\right) \hat{z} - \frac{dB_z}{dy}\hat{y} - \frac{dB_z}{dx}\hat{x}\right]$$

and now, we can apply Gauss' Law for Magnetism to rewrite the $$\hat{z}$$ term as 

$$\vec{F}_\nabla  =  \frac{qr_L^2\omega_L}{2} \left[-\frac{dB_z}{dz} \hat{z} - \frac{dB_z}{dy}\hat{y} - \frac{dB_z}{dx}\hat{x}\right] = -\frac{mv_L^2}{2B}\nabla B_z \approx -\frac{mv_L^2}{2B}\nabla B$$

$$\vec{v}_\nabla = \frac{\vec{F}_\nabla\times\vec{B}}{qB^2} = \frac{mv_L^2}{2qB^3}\vec{B}\times\nabla B$$

This is the force and drift associated with the gradient of the magnetic field.

<h2 id="polar">02 - Derivation of Curvature and Polarization Drift</h2>

To get curvature and polarization drifts, we'll want to decompose the L'Armor-averaged Lorentz equation into a component parallel to and perpendicular to the magnetic field: $$\vec{v}_\text{gc} = \vec{v}_{\text{gc}\perp} + v_{\text{gc}\parallel}\hat{B}$$ and

$$m\frac{d\vec{v}_\text{gc}}{dt} = q\left[\vec{E}(\vec{x}_\text{gc})+ \vec{v}_\text{gc}\times\vec{B}(\vec{x}_\text{gc}) + \langle \vec{v}_L\times(\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\rangle\right]$$

$$m\frac{d(\vec{v}_{\text{gc}\perp} + v_{\text{gc}\parallel}\hat{B})}{dt} = m\frac{d\vec{v}_{\text{gc}\perp}}{dt} + m\frac{d v_{\text{gc}\parallel}}{dt}\hat{B} + mv^2_{\text{gc}\perp}\frac{d\hat{B}}{ds}$$

$$m\frac{d\vec{v}_{\text{gc}\perp}}{dt} + m\frac{d v_{\text{gc}\parallel}}{dt}\hat{B} - mv^2_{\text{gc}\parallel}\hat{B}\cdot\nabla\hat{B} = q\left[\vec{E}(\vec{x}_\text{gc})+ \vec{v}_{\text{gc}\perp}\times\vec{B}(\vec{x}_\text{gc}) + \langle \vec{v}_L\times(\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\rangle\right]$$

Now we can decompose this into an equation for the parallel and perpendicular motions:

$$\text{parallel : } \qquad m\frac{d v_{\text{gc}\parallel}}{dt} = q\vec{E}_\parallel(\vec{x}_\text{gc}) + \langle q\vec{v}_L\times(\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\rangle_\parallel$$

$$\text{perpendicular : } \qquad m\frac{d\vec{v}_{\text{gc}\perp}}{dt} - mv^2_{\text{gc}\parallel}\hat{B}\cdot\nabla\hat{B} = q\vec{E}_\perp(\vec{x}_\text{gc}) + \langle q\vec{v}_L\times(\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\rangle_\perp + \vec{v}_\text{gc}\times\vec{B}(\vec{x}_\text{gc})$$

Now we can rewrite the perpendicular equation as

$$m\frac{d\vec{v}_{\text{gc}\perp}}{dt} = \vec{F}_\perp + \vec{v}_\text{gc}\times\vec{B}(\vec{x}_\text{gc})$$

$$\vec{F}_\perp = mv^2_{\text{gc}\parallel}\hat{B}\cdot\nabla\hat{B} + \langle q\vec{v}_L\times(\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\rangle_\perp + q\vec{E}_\perp(\vec{x}_\text{gc})$$

We're going to break this up into a steady state solution and a first-order perturbation. For the steady state solution, we'll invent a velocity field $$\vec{v}_F$$. This gives us:

$$m\frac{d\vec{v}_{\text{gc}\perp}}{dt} =  0 = \vec{F}_\perp + \vec{v}_F\times\vec{B}(\vec{x}_\text{gc})$$

$$\vec{v}_F = \frac{\vec{F}_\perp \times\vec{B}(\vec{x}_\text{gc})}{qB^2}$$

and now we'll add a first-order perturbation, approximating $$\vec{v}_{\text{gc}\perp} = \vec{v}_F + \vec{v}_P$$. Time variations in $$\vec{v}_P$$ will be negligible in comparison to those of $$\vec{v}_F$$. This leaves us with an expression for the polarization drift of the particle:

$$m\frac{d(\vec{v}_F + \vec{v}_P)}{dt} = m\frac{d\vec{v}_F}{dt} = \vec{F}_\perp + \vec{v}_F\times\vec{B}(\vec{x}_\text{gc}) + \vec{v}_P\times\vec{B}(\vec{x}_\text{gc}) \implies m\frac{d\vec{v}_F}{dt} = \vec{v}_P\times\vec{B}(\vec{x}_\text{gc})$$

$$\vec{v}_P = -\frac{m}{qB^2}\frac{d\vec{v_F}}{dt}\times\vec{B}$$

And finally, the curvature drift comes from considering the acceleration term due to the change of the magnetic field:

$$\vec{F}_R = - mv^2_{\text{gc}\parallel}\hat{B}\cdot\nabla\hat{B} \implies \vec{v}_R = \frac{mv^2_{\text{gc}\parallel}}{qB^2}\vec{B}\times\hat{B}\cdot\nabla\hat{B} = \frac{mv^2_{\text{gc}\parallel}}{qB^3}\vec{B}\times\nabla\hat{B}$$

<h2 id="mirror">03 - Magnetic Mirrors</h2>

We can derive the conditions for magnetic mirrors by considering the parallel equation. Specifically, we can rewrite it in the form of a conservation equation if we multiply the entire thing by $$v_{\text{gc}\parallel}$$. Because we're only considering the parallel component of the electric field, we can write it in terms of a potential $$\varphi$$.

$$m\frac{d v_{\text{gc}\parallel}}{dt} = -q\frac{d\varphi}{ds} + \langle q\vec{v}_L\times(\vec{r}_\text{L}\cdot\nabla)\vec{B}(\vec{x}_\text{gc})\rangle_\parallel$$

$$mv_{\text{gc}\parallel}\frac{d v_{\text{gc}\parallel}}{dt} =  \frac{d}{dt}\left(\frac{mv_{\text{gc}\parallel}^2}{2}\right) = -qv_{\text{gc}\parallel}\frac{d\varphi}{ds} -\frac{mv_L^2}{2B} v_{\text{gc}\parallel}\frac{dB}{ds} = -q\frac{ds}{dt}\frac{d\varphi}{ds} -\frac{mv_L^2}{2B} \frac{ds}{dt}\frac{dB}{ds} = -q\frac{d\varphi}{dt} -\frac{mv_L^2}{2B} \frac{dB}{dt}$$

and so,

$$\frac{d}{dt}\left(\frac{mv_{\text{gc}\parallel}^2}{2} + q\varphi + \mu B\right) = 0 \implies \frac{mv_{\text{gc}\parallel}^2}{2} + q\varphi + \mu B = \text{const}$$

we can see that in the absence of an electromagnetic field, the constant term will just be the total energy of the particle $$W_0$$. If we solve for the parallel velocity, we get:

$$\frac{mv_{\text{gc}\parallel}^2}{2} + q\varphi + \mu B = W_0 \implies v_{\text{gc}\parallel} = \sqrt{\frac{2}{m}(W_0 - \mu B)}$$

We can see that particles with $$W_0 < \mu B$$ will be trapped by the field, whereas particles with $$W_0 > \mu B$$ will escape it. We can derive the condition for magnetic mirroring by involking conservation of $$\mu$$ (which I'll prove later) so that at the point of maximum kinetic energy (and therefore minimum magnetic field), $$\mu = \frac{mv_{L\text{max}}^2}{2B_\text{min}}$$ and so

$$\frac{mv_{L\text{max}}^2}{2B_\text{min}}B_\text{max} = W_0 = \frac{mv_0^2}{2}$$

$$\frac{B_\text{min}}{B_\text{max}} = \frac{v_{L\text{max}}^2}{v_0^2}$$