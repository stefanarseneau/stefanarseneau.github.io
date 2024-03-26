---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: article
title: "corv : compact object radial velocities"
style: assets/css/style.css
---

<a href="https://github.com/vedantchandra/corv">Find corv on github</a>

corv is designed to measure the radial velocities of white dwarfs to the highest possible accuracy. It works by simultaneously fitting several absorption lines to an observed spectrum in order to provide a better measurement of redshift than would otherwise be possible. It operates in two steps:
<ol>
  <li>First, corv generates a rough estimate of $T_{eff}$ and $\log g$ by performing an ordinary least squares regression against the normalized inputted spectrum. These parameters are only rough estimates though. If you need accurate measurements of $T_{eff}$ or $\log g$ then <a href="https://github.com/vedantchandra/wdtools">this is the package you want.</a></li>
  <li>With the best fit spectral parameters identified, corv calculates radial velocity by chi squared minimization on a fine grid. By fitting these parameters in separate steps, radial velocity can be determined much more accurately and with better statistical uncertainties.</li>
</ol>
