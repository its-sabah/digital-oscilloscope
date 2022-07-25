# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:35:18 2019

@author: Sabah
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.widgets as widg #adding widgets
import scipy.signal as sig
plt.close('all')

# Sabah Hussein 14315394
# Scientific Computing: Signal Processing Assingment
    #This code generates either a Sine, Square or Saw-tooth wave, which is 
    #  transformed into a fourier transform. Noise may be added to the signal,
    #   and the user can 'cut' this noise off in the fourier plot.
    #   The inverse fourier transform is then taken of this cut wave and plotted
    #   on a seperate axes.
    #   The user can alter the original signal's phase, time period, frequency,
    #   and number of points.
    # The Inverse fourier transform plotted is incorrect as it disregards half 
    # of the fft. On line 158 I have written and commented out my attempt.

#%% Defining Functions

def sinf(t, freq, phase): # function for sin generator
    return np.sin((freq*2*pi*t) + phase)

def sqrf(t, freq, phase): # function for sqyare wave generator
    return sig.square((freq*2*pi*t) + phase)

def sawtf(t, freq, phase): #function for sawtooth generator
    return sig.sawtooth((freq*2*pi*t) + phase)

def myfft(funct, n, tmax): #function for calculating fourier
    fft = np.fft.fft(funct)
    opow = np.real((np.conj(fft)*fft)/n)
    k = np.arange(1,n/2,1)
    power = opow[1:int(len(k)+1)]
   
    f = (k/tmax) 
    return f, power, fft

def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return array[index]

#%% Defining Variables

pi = np.pi
tmax = 2
freq = 1
n = 100
phase = 0.0
t = np.linspace(0, tmax, n)
y = sinf(t, freq, phase)
yft = myfft(y, n, tmax) # data from fourier transorm

yifft = np.fft.ifft(yft[2]) #inverse fourier
tift = np.linspace(0, tmax, len(yifft))

fullrand = np.random.uniform(low=-1.0, high = 1.0, size=(3000)) #creating a big random

noise = 0
leftcut = 0.5
rightcut = 24.5
cutline = np.linspace(0, 25, 5)
leftline = np.ones(5)*0.5
rightline = np.array([24.5, 24.5, 24.5, 24.5, 24.5])



#%% Defining Callbacks

def update(val):
    #1. getting values from sliders'
    freq = freqHandle.val
    tmax = timeHandle.val
    N = int(pointsHandle.val)
    phase = phaseHandle.val
    noise = noiseHandle.val
    leftcut = leftHandle.val
    rightcut = rightHandle.val
    
    #2. changing the function via radio buttons
    label = buttonHandle.value_selected
    sigdict = {'Sine':1, 'Square':2, 'Saw Tooth':3} #creating a dictionary for functions
    bval = sigdict[label]
    
    t = np.linspace(0, tmax, int(N))
    
    if bval == 1: #radio button dependence
        y = sinf(t, freq, phase) 
    elif bval == 2:
        y = sqrf(t, freq, phase) 
    elif bval == 3:
        y = sawtf(t, freq, phase) 
      
    #3a. calculating noise
    rand = fullrand[0:N]
    if noise == 0:
        y_noise = y
    else:
        y_noise = noise*(y + rand)
    
    #3b. replotting original graph
    oaxHandle.set_xdata(t)
    oaxHandle.set_ydata(y_noise)
    
    oax.set_ylim(max(y_noise)+0.5, min(y_noise)-0.5)
    oax.set_xlim([0, tmax])
    
    #4a. getting fourier values
    yft = myfft(y_noise, N, tmax)
    f1 = yft[0] #frequency 
    power = yft[1] #power
    opower = yft[2] #original power spectrum from fft
    
    #4b. creating cuts
        #left
    if leftcut < min(f1):
        leftcut = min(f1)
        leftline = min(f1)*np.ones(5)
    else: 
        leftline = leftcut*np.ones(5)
        
    leftval = find_nearest(f1,leftcut)
    pmin = int(np.where(f1 == leftval)[0])
    
        #right
    if rightcut > max(f1):
        rightcut = max(f1)
        rightline = rightcut * np.ones(5)
    else: 
        rightline = rightcut*np.ones(5)
    rightval = find_nearest(f1,rightcut)
    pmax = int(np.where(f1 == rightval)[0])
    
    #4c. using cuts for new fourier arrays
    f1m = np.ma.masked_where( f1 < leftcut, f1) #left
    f1_masked = np.ma.masked_where (f1m > rightcut, f1m) #right
    
    #finding peak freuency
#    hfi= f1[np.where(power==max(power))[0] ] 
#    print('Frequency of Fourier', hfi)
    
    #4d. plotting cut lines and arrays
    rightaxHandle.set_xdata(rightline)
    leftaxHandle.set_xdata(leftline)
    fftaxHandle.set_xdata(f1_masked)
    fftaxHandle.set_ydata(power)
    fftax.set_ylim(0,max(power)+20)
    
    #5a. calculating inverse fourier transforms
    power_cut = opower[pmin: pmax] #applying cuts
    yifft = np.fft.ifft(power_cut) #y inverse fourier
    tift = np.linspace(0, tmax, len(yifft)) #time cuts
    
    #The way I have calculated the inverse fourier transform is incorrect, as I've disgarded half of the information. 
    #I have attempted to correct this, but have been getting shape errors I don't understand.
    # As a result, I have commented out the attempt:
#    fft_copy = np.copy(opower)
#    
#       #creating postive and negative cuts for the two peaks
#    if N & 1:#odd
#        fft1 = fft_copy[0:int((N+1)/2)]
#        fft2 = fft_copy[int((N+1)/2):N]
#    else: #even
#        fft1 = fft_copy[0:int(N/2)]
#        fft2 = fft_copy[int(N/2):0] 
#    
#    #finding the negative and positive peaks
#    ffta = fft1[lowerfouriercut:higherfouriercut]
#    fftb = fft2[higherfouriercut:(n-lowerfouriercut)]
#    final_fourier = np.add(ffta, fftb) #adding them together
#    ioaxHandle.set_ydata(final_fourier)
    
    
    #5b. plotting ifft
    ioaxHandle.set_xdata(tift)
    ioaxHandle.set_ydata(yifft)
    ioax.set_xlim(0,max(tift))
    ioax.set_ylim(min(yifft)-0.5,max(yifft)+0.5)

    plt.draw()
 
def closeCallback(event): #Callback for closing figure
    plt.close('all')
    
fig = plt.figure(figsize=(18, 8))

    #original axes for signal
oax = plt.axes([0.29, 0.76, 0.6, 0.20])
oaxHandle, = plt.plot(t, y, lw = 2, color = 'red') #sets signal 
#oax.autoscale(enable=True, axis='x', tight=None)
oax.set_xlabel('Time ($s$)')
oax.set_ylabel('Signal ($V$)')
oax.set_title('Original Signal', loc='right', fontweight="bold")

    #fourier transform axes for signal
fftax = plt.axes([0.29, 0.41, 0.6, 0.29])
f1 = yft[0]
power = yft[1]
fftaxHandle, = plt.plot(f1, power, lw = 2, color = 'blue')
fftax.set_xlabel('Frquency ($Hz$)')
fftax.set_ylabel('Power ($V^2$)')
fftax.set_title('Fourier Transform', loc='right', fontweight="bold")

    #inverse fourier axes
ioax = plt.axes([0.29, 0.06, 0.6, 0.29])
ioaxHandle, = plt.plot(tift, yifft, 'k')
ioax.set_xlabel('Time ($s$)')
ioax.set_ylabel('Signal ($V$)')
ioax.set_title('Inverse Fourier Transform', loc='right', fontweight="bold")

    #creating axes for cut on fourier graph
leftax = fftax.twinx()
leftaxHandle, = plt.plot(leftline, cutline, 'y--')
rightax = fftax.twinx()
rightaxHandle, = plt.plot(rightline, cutline, 'y--')
    
    #frequency slider
fax = plt.axes([0.93, 0.15, 0.02, 0.7])
freqHandle = widg.Slider(fax, 'Frequency ($Hz$)', 0.1, 5, valinit = freq, orientation='vertical')
freqHandle.on_changed(update)

    #time slider
tax = plt.axes([0.05, 0.35, 0.04, 0.35])
timeHandle = widg.Slider(tax, 'Time ($s$)', 1, 30, valinit = tmax, orientation='vertical')
timeHandle.on_changed(update)

   #phase slider
pax = plt.axes([0.11, 0.35, 0.04, 0.35])
pi = np.pi
phaseHandle = widg.Slider(pax, 'Phase ($rad$)', 0, (2*pi), valinit = 0, orientation='vertical')
phaseHandle.on_changed(update)

    #points slider
nax = plt.axes([0.17, 0.35, 0.04, 0.35])
pointsHandle = widg.Slider(nax, 'Points', 8, 200, valinit = n, valstep = 1, orientation='vertical')
pointsHandle.on_changed(update)

    #close button  
bax = plt.axes([0.06, 0.77, 0.12, 0.12])
buttonHandle = widg.RadioButtons(bax, ('Sine', 'Square', 'Saw Tooth'))
buttonHandle.on_clicked(update)

    #noise
nax = plt.axes([0.05, 0.25, 0.17, 0.04])
noiseHandle = widg.Slider(nax, 'Noise', 0, 3, valinit = noise, orientation='horizontal')
noiseHandle.on_changed(update)

    #left cut
lax = plt.axes([0.05, 0.19, 0.17, 0.04])
leftHandle = widg.Slider(lax, 'Left Cut', 0, 10, valinit = leftcut, valstep = 0.1, orientation='horizontal')
leftHandle.on_changed(update)

    #right cut
rax = plt.axes([0.05, 0.13, 0.17, 0.04])
rightHandle = widg.Slider(rax, 'Right Cut', 1, 25, valinit = rightcut, valstep = 0.1, orientation='horizontal')
rightHandle.on_changed(update)

    #close button
cax = plt.axes([0.09, 0.03, 0.07, 0.07])
closeHandle = widg.Button(cax, 'Close')
closeHandle.on_clicked(closeCallback)   
