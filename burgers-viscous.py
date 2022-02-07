

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sys
from matplotlib import colors
from IPython import embed
from os import path

# We define inverese and direct fft reversely, because the renormalisation
# makes more sense: the fft should be an integral and thus have a factor 1/n in
# front of the noise.
from numpy.fft import fft as ifft
from numpy.fft import ifft as fft

class burgers:
	"""
	We model Burgers' equation via a Galerkin approximation and Neumann
    boundary conditions (so basis of cosines on the interval [0,2 pi])
	"""
	def __init__(self):

		# Initial state of the system in terms of Fourier coefficients:
		# This is the number of eigenfunctions we use (for the definitions
		# below to work correctly with the Numpy implementation of the fast
		# Fourier transform this number should be odd: zeroth + first half positive,
		# + second half negative modes)
		self.N   = 1601
		# And we set the time discretization for our problem
		self.dt  = 0.0003
		# And we set the parameter a for the hyperviscosity (a =1 means
		# Laplacian: -(-Delta)^a)
		self.a   = 1.0

		# We define the current value in real coordinates
		self.value1 = 2*np.ones(shape = (self.N))
		self.value1[self.N//2:self.N] = 0

		self.value2 = 2*np.ones(shape = (self.N))
		self.value2[0:self.N//2] = 0

		# And on a coarser scale the value for the picture
		self.V = 701
		self.visual1 = np.ones(shape = (self.V))
		self.visual2 = np.ones(shape = (self.V))

		# The initial condition in its Fourier coefficients 
		self.state1 = fft(self.value1)
		self.state2 = fft(self.value2)

		# Here we store the forces (the nonlinearity and the noise)
		self.nonli1 = np.zeros(shape = (self.N), dtype = complex)
		self.nonli2 = np.zeros(shape = (self.N), dtype = complex)
		self.noise = np.random.normal(loc = 0.0, scale =1.0, size =self.N)

		# We define the gradient as a multiplier in Fourier coordinates
		self.grad  = np.zeros(shape = (self.N), dtype = complex)
		for i in range(1,self.N//2+1):
			self.grad[i] = complex(0.0, -1.0*i)
		for i in range(self.N//2 + 1, self.N):
			self.grad[i] = complex(0.0, 1.0*(i-self.N//2))

		# We define the resolvent of the fractional Laplacian as a multiplier
		# in Fourier coordinates
		self.relap  = np.ones(shape = (self.N), dtype = complex)
		for i in range(1,self.N//2+1):
			self.relap[i] = complex(1.0/(1.0+(i**(2*self.a))*self.dt), 0.0)
		for i in range(self.N//2 + 1, self.N):
			self.relap[i] = complex(1.0/(1.0+((i-self.N//2)**(2*self.a))*self.dt), 0.0)

	def evaluate(self):
		# This function adjourns the value of the real state of the system.
		self.value1 = ifft(self.state1, self.N).real
		self.value2 = ifft(self.state2, self.N).real

	def visualize(self):
		# This function adjourns the value of the visualizer
		self.visual1 = ifft(self.state1, self.V).real
		self.visual2 = ifft(self.state2, self.V).real

	def nonlinearity(self):
		# We define the nonlinearity d_x u^2 in Fourier coefficients
		self.evaluate()
		self.nonli1 = np.multiply(self.grad, fft(self.value1**2, self.N), dtype = complex)
		self.nonli2 = np.multiply(self.grad, fft(self.value2**2, self.N), dtype = complex)

	def renoise(self):
		# We adjourn the noise
		self.noise = 0.5*np.random.normal(loc = 0.0, scale =1.0, size
								=self.N)*np.sqrt(self.N*self.dt)

	def solver(self):
		# We do one more step in the implicit Euler approximation

		# We start by computing the nonlinearity and the noise
		self.nonlinearity()
		self.renoise()

		self.state1 = np.multiply(self.relap, self.state1 +
							self.dt*(self.nonli1)+ fft(self.noise), dtype = complex)
		self.state2 = np.multiply(self.relap, self.state2 +
							self.dt*(self.nonli2)+ fft(self.noise), dtype = complex)

def animate(i):

	global bu, space_pts, ax, fig, time_text

	# Real time is:
	ani_time = i*bu.dt

	# Redefine the plot
	bu.visualize()
	lines_a.set_data(space_pts, np.fft.fftshift(bu.visual1))
	lines_b.set_data(space_pts, np.fft.fftshift(bu.visual2))

	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time))

	# We print the step we are in
	sys.stdout.flush()
	sys.stdout.write("\r Step = {}".format(i))

	# And we do the next step:
	bu.solver()
	return [lines_a,] + [lines_b,] + [time_text,]

bu = burgers()
space_pts = np.linspace(-np.pi, np.pi, bu.V)

# We set up the picture
fig        = plt.figure()
ax         = plt.axes(xlim=(-np.pi -0.2, np.pi+0.2), ylim = (-5.0, 5.0))
time_text  = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
lines_a,   = ax.plot([],[], lw = 1.3)
lines_b,   = ax.plot([],[], lw = 1.3)
plt.title("Hyperviscous Burgers'")

# We let the animation go.
ani = FuncAnimation(fig, animate, frames= 20000, repeat=False)
mywriter = animation.FFMpegWriter(fps=20, bitrate=60000, extra_args=['-pix_fmt', 'yuv420p'])
ani.save('burgers.mp4',writer=mywriter)
