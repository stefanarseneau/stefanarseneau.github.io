---
title: "compact object radial velocities"
layout: post
date: 2024-03-31 23:16
image: /assets/images/markdown.jpg
headerImage: false
tag:
- markdown
- elements
star: true
category: blog
author: johndoe
description: Markdown summary with different options
---

I figure it's probably a good idea to jot down some notes about [corv](https://github.com/vedantchandra/corv). Probably this'll wind up becoming part of the README on the github 
(once I make it all professional). 

Basically, corv is for measuring radial velocities of white dwarfs. Right now, it only works on DAs but we're working on adding support for DBs too. It works by taking [template
WD spectra](https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/) and simultaneously fitting an arbitrary number of Balmer lines to measure a radial
velocity. This lets us get a more accurate measurement of the target's radial velocity.
