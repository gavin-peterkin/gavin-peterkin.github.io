---
layout: post
title:  "Experimental Astronomy Lab Reports"
date:   2017-07-13 16:38:14 -0700
categories: Science
<!-- featured-img: /images/mandelbrot_banner.png -->
---
# Intro

This post will include links to some of my experimental astronomy lab work,
which I completed in the senior year of my undergraduate career. I've roughly
organized them in order from most to least interesting! I used GNU
Octave (the free version of Matlab) for all of the data analysis. Below each link
you'll see the quoted abstract from the report.

# Links to the work

* <a href="/file_content/HR_diagram.pdf">Homemade HR Diagram</a>:
  Producing a Hertzprung-Russell diagram that plots a stars brightness with
  respect to its temperature.

  > In this lab, we image the globular cluster M15 (NGC 7078) using a CCD camera attached
to the James R. Houck 25-inch Telescope with three different Johnson-Cousins-Glass filter
bands (B, V, and R). We also image the standard star SA 115 427 in the same three bands
to calibrate the photometric data. Using these data, I create three calibrated Hertzprung
Russell diagrams (V vs B-V, R vs V-R, and R vs B-R). These diagrams are used to determine
the age, metallicity, and distance of the cluster by comparison with theoretical
isochrone models from Leo Girardi’s Padova Database. The age is found to be ~12 Gya while
the distance is ~12.6 kPc.

* <a href="/file_content/galaxy_kinematics.pdf">Galaxy Kinematics</a>
  Studying the motion of a galaxy using observed spectroscopic data.

  >In this lab, we use a long-slit spectrograph (600 grooves/mm, 135mm f.l., 200μm slit,
12d) on the James R. Houck 25-inch Telescope to study the spiral galaxy NGC 7448. More
specifically, I determine the redshift of the galaxy and study its rotation. From the
redshift, I determine the distance to the galaxy using a value of Hubble’s constant from
the literature. From the tilt of the H-α line, I determine the rate of rotation and find
a value for galactic mass gathered from the kinematics of the galaxy itself (kinematic
mass). Using results from the literature, I also determine luminous mass. Finally, it’s
found that the kinematic mass is about five times greater than the luminous matter. This
result and it’s meaning is discussed.

* <a href="/file_content/HI_spectroscopy_copy.pdf">Milky Way HI Spectroscopy</a>
  Studying the radio spectrum of the Milky Way in order to see rotation.

  >In this lab, we use the previously studied 3.8 meter parabolic Josephine Hopkins Radio
  Telescope to obtain spectra centered on 1420 MHz with a 10 MHz band at several galactic
  longitudes between l = 0° and l = 90°. After performing a thorough analysis of these
  data, which takes into account the local standard of rest (LSR) corrections among other
  intricacies, I use the results to determine the respective minima and maxima Doppler
  velocities at three different longitudes. From these results and assuming a flat
  rotation curve, it is possible to find the approximate radius (R0) of HI gas in the
  galaxy, which is determined to be ~16 kpc, and the value of the flat rotational
  velocity (V0), which is found to be ~280 km/s. Finally, these results are compared to
  previous results from the literature.

* <a href="/file_content/radiotelescope_copy.pdf">Radiometer Characterization</a>
  Detailed analysis of the capabilities of a radio telescope.

  >This paper details some of the very basic characteristics of the 3.8 meter parabolic
  radio telescope atop the Space Sciences Building located on Cornell's Ithaca campus.
  Through the use of an absorber material, which is held directly in front of the helical
  receiver, a noise diode, which is capable of injecting noise into the analog signal
  path prior to any processing, and manipulation of the attenuation, I calculate the
  system temperature and the effective temperature of the calibration diode. In order to
  determine the beam width of the antenna, a drift scan is performed on the sun, which
  has a well-known angular size. I use the previously determined system temperature and
  the same drift-scan data to estimate the effective temperature of the sun. The system
  temperature is found to be about 160K and the calibration temperature is around 860K.
  The experimentally determined beam width is about 4 degrees (14 deg2).

* <a href="/file_content/CCD_characterization.pdf">CCD Characterization</a>
  Determining whether or not a lab-grade CCD camera meets manufacturer's
  specifications.

  >This paper details the performance characteristics of a 2048x512 pixel array charge-
  coupled device (CCD) camera, model DU440A-BV, manufactured by Andor Technology for
  applications in spectroscopy and imaging. The back-illuminated CCD chip is the CCD42-10bi
  manufactured by Marconi EEV. Among the characteristics studied in a controlled laboratory
  environment are: read noise, dark current, linearity, blooming, and charge transfer
  efficiency. The CCD was set to a pixel readout speed of 50kHz and thermoelectrically
  cooled to -40°C. Read noise is determined to be ~4e-s. Dark current hovered just below
  detection, the response was extremely linear for non-extreme exposure durations (>99%),
  the peak charge storage was determined to be about 120,000 e-s, and charge transfer
  efficiency was found to have a lower limit of 99%.
