---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: article
title: "corv : compact object radial velocities"
style: assets/css/style.css
page_image: assets/corv.png
---

<a href="https://github.com/vedantchandra/corv">Find corv on github</a>

corv is designed to measure the radial velocities of white dwarfs to the highest possible accuracy. It works by simultaneously fitting several absorption lines to an observed spectrum in order to provide a better measurement of redshift than would otherwise be possible. It operates in two steps:
<ol>
  <li>First, corv generates a rough estimate of \(T_{eff}\) and \(\log g\) by performing an ordinary least squares regression against the normalized inputted spectrum. These parameters are only rough estimates though. If you need accurate measurements of \(T_{eff}\) or \(\log g\) then <a href="https://github.com/vedantchandra/wdtools">this is the package you want.</a></li>
  <li>With the best fit spectral parameters identified, corv calculates radial velocity by chi squared minimization on a fine grid. By fitting these parameters in separate steps, radial velocity can be determined much more accurately and with better statistical uncertainties.</li>
</ol>

**How Do I Use corv?**

Several examples of corv in action can be found <a href="https://github.com/stefanarseneau/corvtutorial">here.</a> To fit a radial velocity, the first thing you need to do is tell corv what lines to fit and what to fit it with. This is done by creating a corv model. Lines can be fit using either Voigt profiles or actual DA white dwarf model spectra. To fit with a Voigt profile fitting the Balmer alpha, beta, gamma and delta lines, one might call:

<code>corvmodel = corv.models.make_balmer_model(nvoigt = 2, names = ['a','b','g','d'])</code>

Likewise, if you wanted to fit a corv model on the same lines using the <a href="https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/">Warwick DA model spectra,</a> you'd call:

<code>corvmodel = corv.models.make_warwick_da_model(names = ['a','b','g','d'])</code>

corv uses the 3D NLTE DA model spectra provided at the above link.
