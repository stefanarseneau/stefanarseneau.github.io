---
layout: note
title:  "Inverse Methods"
---

Notes based on [Inverse Problems: Basics, Theory and Applications in Geophysics](https://link.springer.com/book/10.1007/978-3-319-48384-9) by Mathias Richter and [Inverse Theory and Applications in Geophysics](https://www.sciencedirect.com/book/9780444626745/inverse-theory-and-applications-in-geophysics) by Michael S. Zhdanov.

<p>
    <h2>Table of Contents ---</h2>
<list>
    <li><a href="#examples">01 - Examples of inverse problems, Fredholm and Volterra equations</a></li>
</list>
<hr>
</p>

<h2 id="examples">01 - Examples of inverse problems, Fredholm and Volterra equations</h2>

Growth rates for things like bacterial colonies are governed by the initial value problem 

$$w'(t) = u(t)w(t), \text{ } w(t_0) = w_0 > 0$$ 

on the interval $$t_0 \geq t \geq t_1$$. For a given continuous function $$u: [t_0, t_1]\to \mathcal{R}$$, which is the growth rate of the problem, the equation has a unique continuously differentiable solution $$w: [t_0, t_1]\to [0, \infty)$$: 

$$w(t) = w_0\exp(U(t)) \text{, } U(t) = \int_{t_0}^{t_1} u(s) ds.$$

This means that the *cause*, which is the growth rate $$u(t)$$, implies an *effect* on the population size $$w(t)$$. We can describe this as a mapping $$T: u\to w$$ parameterized by $$t_0, t_1, w_0$$. The associated inverse problem is the construction of a function such that for a given $$w(t)$$, $$T^{-1}(w) = u$$. This can be solved explicitly: 

$$w'(t) = u(t)w(t) \implies u(t) = \frac{d}{dt} \ln w(t).$$ 

In practice though, we have only a finite amount of data, so the best we can do is approximate $$u(t)$$ (if we're solving the direct problem) or $$w(t)$$ (if we're solving the inverse problem).

For the direct problem this is not an issue. Taking $$\hat{u}$$ to be an approximator of $$u$$ such that 

$$\max\{|u(t) - \hat{u}(t)|\}\leq \varepsilon, \varepsilon > 0,$$ 

we can find that 

$$\max\{|u(t) - \hat{u}(t)|\}\leq C\varepsilon.$$ 

This means that up to a constant, variations can be mitigated by taking better measurements. 

If we're interested in solving the inverse problem, we can add a perturbation to the measured signal $$w(t)$$ such that 

$$\hat{w}(t) = w_n(t) = w(t) * \left(1 + \frac{1}{\sqrt{n}}\cos(nt)\right)$$ 

and we can see that $$\hat{w}\to w$$ as $$n \to \infty$$ which means that successively higher $$n$$ provide a better and better approximation for $$w$$.

Now we can actually try to implement this in python.
<pre><code class="python">
import jax.numpy as np
from jax import grad, vmap
import matplotlib.pyplot as plt

w = lambda t: np.exp(np.sin(t)) # unperturbed signal
w_n = lambda t, n: w(t) * (1 + (1 / np.sqrt(n)) * np.cos(n*t)) # perturbed signal

def w_inverse(t):
    ln_w = lambda t: np.log(w(t))
    return vmap(grad(ln_w))(t)

def wn_inverse(t, n):
    ln_w = lambda t: np.log(w_n(t, n))
    return vmap(grad(ln_w))(t)
</code></pre>

We can see that as the perturbation is increased and the measurement more accurately reflects the signal, the calculated inverse function starts to diverge. In fact we can prove that despite the fact that 

$$ \max\{|w_{n}(t) - w(t)|\} \to w \text{ as } n \to \infty,$$

mapping that to some inverse signal $$u_n$$ leads to 

$$\max\{|u_n(t) - u(t)|\} \to \infty \text{ as } n\to \infty.$$

This is because of the differentiation operation. Integrating acts to smooth out the signal, so it makes sense that its inverse, differentiation, would roughen it. This will blow up the perturbations in the input, which will make the explicit inverse equation almost useless in practice.

<pre><code class="python">
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13,4))

ax1.set_title('Input')
ax1.plot(t, w_n(t, n_1), c = 'r', alpha=1, label = f'perturbed, n={n_1}')
ax1.plot(t, w_n(t, n_2), c = 'green', alpha=1, label = f'perturbed, n={n_2}')
ax1.plot(t, w(t), c = 'k', label = f'true signal')
ax1.legend(framealpha=0)
ax1.set_ylim(-8,8)

ax2.set_title('Output')
ax2.plot(t, wn_inverse(t, n_1), c = 'r', alpha=1, label = f'perturbed, n={n_1}')
ax2.plot(t, wn_inverse(t, n_2), c = 'green', alpha=1, label = f'perturbed, n={n_2}')
ax2.plot(t, w_inverse(t), c = 'k')
ax2.set_ylim(-8,8)
</code></pre>

<img itemprop="image" src="/images/inverse_problem_01.png">

**Fredholm and Volterra Equations**

Suppose we have some sort of integral equation which maps some cause to some effect, $$u, w:[a,b]\to \mathcal{R}$$, and has the form

$$\int_a^b k(s, t, u(t)) dt = w(s) \text{ , } a \leq s \leq b$$

where $$k:[a,b]^2\times \mathcal{R}\to \mathcal{R}$$ is a kernel function. An equation of this form is called a *Fredholm integral equation of the first kind.* A special case of this is the *linear* Fredholm equation of the first kind, which has form

$$\int_a^b k(s,t) u(t) dt = w(s)\text{ , } a \leq s \leq b.$$

If this kernel has the special property that $$k(s,t) = 0$$ when $$t > s$$, then the equation becomes a *Volterra integral equation of the first kind*, which can be written as

$$\int_a^s k(s,t) u(t)dt = w(s)\text{ , } a\leq s\leq b.$$

Another special case is that in which the kernel function has the property $$k(s,t) = k(s - t)$$. In this case, the linear Fredholm equation is called a *convolutional equation*:

$$\int_a^b k(s-t)u(t) dt = w(s)\text{ , } a \leq s \leq b.$$

Finally, we have Volterra or Fredholm equations of the second kind when the function $$u(t)$$ appears both inside and outside of the integral:

$$u(s) + \lambda\int_a^b k(s,t)u(t)dt = w(s)\text{ , }a \leq s \leq b\text{ , } \lambda\in\mathcal{R}.$$

Linear Fredholm equations of the first and second kind have substantially different properties. Most notably, if the kernel $$k$$ is smooth (in the mathematical sense, e.g. continuous), then the mapping $$u\to w$$ also has that smoothing property. This means that a solution which involves inverting that mapping, which must necessarily invert an integration, must necessarily roughen $$w$$ and amplify the error. The computation of derivatives in the population growth example is equivalent to solving a Volterra integral equation:

$$u(t) = w'(t)\text{, }w(t_0) = 0 \iff w(t) = \int_{t_0}^t u(s) ds.$$

Integral equations of the second kind contain unsmoothed versions outside of the integral, so they don't necessarily have to roughen up in the inverse.