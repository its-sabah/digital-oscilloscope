# Digital Oscilloscope
Objective: GUI that allows users to investigate Fourier waves and their inverse. 

## Background and Motivation: 
As a required submission to the Scientific Computing module as part of my degree, I created a model that allows users to investigate how different wave types and varying inputs effect Fourier Transforms & their inverses.

The GUI allows users to change the original signal, with control over: wave type, time (s), phase (rad), frequency (Hz), noise and number of points.

This then changes the Fourier Transform and Inverse Fourier Transform respectively. Users can apply left and right cuts to the Fourier transform; the more information cut from this signal, the less the inverse graph resembles the original signal.

Graded at 85%.

## How to Run
To generate the GUI, follow the following steps:
1. Ensure that Anaconda/Spyder is downloaded with the standard packages (numpy, matplotlib, scipy)
2. Download this code
3. Enable interactive graphs by changing the backend to automatic:
  a. Tools > preferences > IPython console > Graphics > Graphics backend > Backend: Automatic
4. Click run & have fun!

![gui screenshot](https://i.imgur.com/TnCE3Du.jpeg)


## Resources
https://docs.spyder-ide.org/current/installation.html

https://en.wikipedia.org/wiki/Fourier_transform

https://en.wikipedia.org/wiki/Fourier_inversion_theorem

https://stackoverflow.com/questions/23585126/how-do-i-get-interactive-plots-again-in-spyder-ipython-matplotlib

