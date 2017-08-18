---
layout: post
title:  "Fractals Revisited with TensorFlow"
date:   2017-08-16 16:38:14 -0700
categories: Science
---

# Introduction

After learning more about _Tensorflow_ in class the other day, I decided it would
be really easy and fun to revisit some of the work I did with fractals in a
[previous post]({% post_url 2017-07-18-fractals %})
but using the excellent [Tensorflow library](https://www.tensorflow.org) instead.
This will be an incredibly short post.

As I learned, tensorflow can be useful for more than just constructing neural networks
although that really is its primary purpose. It can be used for tensor operations
of any kind...including things to do with fractals.

I should also note that this post was inspired by a particular lab exercise from my
work at Galvanize.

# Some basic imports and setup

```python
from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib import animation
from matplotlib import colors

%matplotlib inline
plt.style.use('bmh')
c_map = plt.get_cmap('Dark2')
```

# Complex number plane
```python
real_part = np.arange(start=-2, stop=2.05, step=0.005)
imag_part = np.arange(start=-2, stop=2.05, step=0.005)

X, Y = np.meshgrid(real_part, imag_part)
Z = X + 1j*Y
```

# Starting the tensorflow session

```python
sess = tf.InteractiveSession()

zs = tf.Variable(Z, dtype=tf.complex64)

c = tf.constant(-0.4 + 0.6j, dtype=tf.complex64)
horizon = tf.constant(4., dtype=tf.float32)

ns = tf.Variable(np.zeros_like(Z), dtype=tf.int32)
```

# The meat of the problem

The Julia set is defined as the set of complex numbers that do not converge to
a limit after a mapping is repeatedly applied. In this case the mapping I'll use
is the very common $f(z) = z^2 + C$ although in principle you could use any function.

I can define a tensorflow operation that will take z, calculated the next value of z
and update, and test for convergence. Not only that--I can do that in one simple
"group" in tensorflow.

```python
step = tf.group(
    tf.assign(zs, tf.add(tf.multiply(zs, zs), c)),
    tf.assign_add(ns, tf.cast(tf.less(tf.abs(zs), horizon), tf.int32))
)
```

# Run iterations
```python
tf.global_variables_initializer().run()

for i in range(100):
    step.run()
```

# Plot and enjoy!

```python
def display_fractal(ns):
    """Display an array of iteration counts as a
       colorful picture of a fractal."""
    ns_cyclic = (6.28*ns/20.0).reshape(list(ns.shape) + [1])
    color_channels = [10 + 20*np.cos(ns_cyclic),
                      30 + 50*np.sin(ns_cyclic),
                      155 - 80*np.cos(ns_cyclic)]
    img = np.concatenate(color_channels, 2)
    # Color the points that never escape black.
    img[ns == ns.max()] = 0
    # Keep the color channels between 0 and 255 (RGB).
    img = np.uint8(np.clip(img, 0, 255))
    return img

fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(display_fractal(ns.eval()));
ax.set_xticklabels('');
ax.set_yticklabels('');
ax.grid()
```

![png](/images/fractal2/output_6_0.png)
