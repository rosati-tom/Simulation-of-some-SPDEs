

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sys
from matplotlib import colors
from IPython import embed
from os import path

import time
from functools import wraps

# We define inverese and direct fft reversely, because the renormalisation
# makes more sense: the fft should be an integral and thus have a factor 1/n in
# front of the noise.
from numpy.fft import fft2 as ifft
from numpy.fft import ifft2 as fft


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class vorticity:
	"""
	We simulate the vorticity equation with a shear force
	"""
	def __init__(self):

		# Initial state of the system in terms of Fourier coefficients:
		# This is the number of eigenfunctions we use (for the definitions
		# below to work correctly with the Numpy implementation of the fast
		# Fourier transform this number should be odd: zeroth + first half positive,
		# + second half negative modes)
		self.N   = 1001
		# And we set the time discretization for our problem
		self.dt  = 0.01
		# And we set the parameter a for the viscosity
		self.nu  = 0.005
		
		# We define our two initial conditions.
		self.space = np.linspace(-0.5, 0.5, self.N)
		self.X, self.Y = np.meshgrid(self.space, self.space)
		
		self.vort1 = np.zeros(shape = (self.N,self.N), dtype = float)
		self.vort2 = np.zeros(shape = (self.N,self.N), dtype = float)
	
		self.vort1 = np.sin(2*np.pi*self.X)
		self.vort2 = np.sin(2*np.pi*self.Y)
		
		# And we define the noise, which we immediately update
		self.noise = np.zeros(shape = (self.N,self.N), dtype = float)
		self.renoise()

		# And on a coarser scale the value for the picture
		self.V = 100
		self.visual1 = np.zeros(shape = (self.V,self.V), dtype = float)
		self.visual2 = np.zeros(shape = (self.V,self.V), dtype = float)

		# The initial condition in its Fourier coefficients 
		self.vort1ft = fft(self.vort1)
		self.vort2ft = fft(self.vort2)
		
		# Placeholders for the gradients
		self.grad1_x = np.zeros(shape = (self.N,self.N))
		self.grad1_y = np.zeros(shape = (self.N,self.N))
		
		self.grad2_x = np.zeros(shape = (self.N,self.N))
		self.grad2_y = np.zeros(shape = (self.N,self.N))
		
		# Placeholders for the stream function
		self.stream1_x = np.zeros(shape = (self.N,self.N))
		self.stream1_y = np.zeros(shape = (self.N,self.N))
		
		self.stream2_x = np.zeros(shape = (self.N,self.N))
		self.stream2_y = np.zeros(shape = (self.N,self.N))

		# We define the inverse Laplacian as a multiplier
		# This leaves the zeroth Fourier mode unchanged
		self.invlaplace = np.zeros(shape = (self.N, self.N), dtype= complex)
		for i in range(0,self.N):
			for k in range(0, self.N):
				if i>0 or k > 0:
					if i <self.N//2+1 and k <self.N//2+1:
						self.invlaplace[i,k] = complex(- 1/(float(i)**2 + float(k)**2),0.0)
					if i <self.N//2+1 and  k >= self.N//2+1:
						self.invlaplace[i,k] = complex(- 1/(float(i)**2 + (self.N-k)**2),0.0)
					if  i >= self.N//2+1 and  k < self.N//2+1:
						self.invlaplace[i,k] = complex(- 1/(float(self.N-i)**2 + float(k)**2),0.0)
					if i >= self.N//2+1 and  k >= self.N//2+1:
						self.invlaplace[i,k] = complex(- 1/(float(self.N-i)**2 + float(self.N-k)**2),0.0)
						
		# This is the multiplier for differentiation in the x variable
		self.nabla_x = np.zeros(shape = (self.N, self.N), dtype = complex)
		for i in range(0,self.N):
			for k in range(0, self.N):
				if i>0 or k > 0:
					if i <self.N//2+1 and k <self.N//2+1:
						self.nabla_x[i,k] = complex(0.0, 1.0*i)
					if i <self.N//2+1 and  k >= self.N//2+1:
						self.nabla_x[i,k] = complex(0.0, 1.0*i)
					if  i >= self.N//2+1 and  k < self.N//2+1:
						self.nabla_x[i,k] = complex(0.0, -1.0*(self.N-i))
					if i >= self.N//2+1 and  k >= self.N//2+1:
						self.nabla_x[i,k] = complex(0.0, -1.0*(self.N-i))
		
		# This is the multiplier for differentiation in the y variable
		self.nabla_y = np.zeros(shape = (self.N, self.N), dtype = complex)
		for i in range(0,self.N):
			for k in range(0, self.N):
				if i>0 or k > 0:
					if i <self.N//2+1 and k <self.N//2+1:
						self.nabla_y[i,k] = complex(0.0, 1.0*k)
					if i <self.N//2+1 and  k >= self.N//2+1:
						self.nabla_y[i,k] = complex(0.0, -1.0*(self.N-k))
					if  i >= self.N//2+1 and  k < self.N//2+1:
						self.nabla_y[i,k] = complex(0.0, 1.0*k)
					if i >= self.N//2+1 and  k >= self.N//2+1:
						self.nabla_y[i,k] = complex(0.0, -1.0*(self.N-k))
						
		# This is the Fourier multiplier for the x derivative of the stream function
		self.invlaplace_x = np.ones(shape = (self.N), dtype = complex)
		self.invlaplace_x = np.multiply(self.invlaplace, self.nabla_x, dtype = complex)
		
		# This is the Fourier multiplier for the x derivative of the stream function
		self.invlaplace_y = np.ones(shape = (self.N), dtype = complex)
		self.invlaplace_y = np.multiply(self.invlaplace, self.nabla_y, dtype = complex)

		# We define the resolvent of the fractional Laplacian as a multiplier
		# in Fourier coordinates
		self.relap  = np.zeros(shape = (self.N, self.N), dtype = complex)
		for i in range(0,self.N):
			for k in range(0, self.N):
				if i>0 or k > 0:
					if i <self.N//2+1 and k <self.N//2+1:
						self.relap[i,k] = complex(1.0/(1.0+(self.nu*i**2+self.nu*k**2)*self.dt), 0.0)
					if i <self.N//2+1 and  k >= self.N//2+1:
						self.relap[i,k] = complex(1.0/(1.0+(self.nu*i**2+self.nu*(self.N -k)**2)*self.dt), 0.0)
					if  i >= self.N//2+1 and  k < self.N//2+1:
						self.relap[i,k] = complex(1.0/(1.0+(self.nu*(self.N-i)**2+ self.nu*k**2)*self.dt), 0.0)
					if i >= self.N//2+1 and  k >= self.N//2+1:
						self.relap[i,k] = complex(1.0/(1.0+(self.nu*(self.N-i)**2+ self.nu*(self.N-k)**2)*self.dt), 0.0)
		
	def force(self):
		
		# We compute the gradient
		self.grad1_x = ifft(np.multiply(self.nabla_x, self.vort1ft, dtype = complex)).real
		self.grad1_y = ifft(np.multiply(self.nabla_y, self.vort1ft, dtype = complex)).real
		
		self.grad2_x = ifft(np.multiply(self.nabla_x, self.vort2ft, dtype = complex)).real
		self.grad2_y = ifft(np.multiply(self.nabla_y, self.vort2ft, dtype = complex)).real
		
		# We compute the stream function
		self.stream1_x = ifft(np.multiply(self.invlaplace_x, self.vort1ft, dtype = complex)).real
		self.stream1_y = ifft(np.multiply(self.invlaplace_y, self.vort1ft, dtype = complex)).real
		
		self.stream2_x = ifft(np.multiply(self.invlaplace_x, self.vort2ft, dtype = complex)).real
		self.stream2_y = ifft(np.multiply(self.invlaplace_y, self.vort2ft, dtype = complex)).real
		
		# We compute the force in Fourier coordinates
		self.force1 = fft(np.multiply(self.stream1_x, self.grad1_x, dtype = complex) - np.multiply(self.stream1_y, self.grad1_y, dtype = complex))
		
		self.force2 = fft(np.multiply(self.stream2_x, self.grad2_x, dtype = complex) - np.multiply(self.stream2_y, self.grad2_y, dtype = complex))
		
		
	def evaluate(self):
		# This function adjourns the value of the real state of the system.
		self.vort1 = ifft(self.vort1ft, s = (self.N, self.N)).real
		self.vort2 = ifft(self.vort2ft, s = (self.N, self.N)).real

	def visualize(self):
		# This function adjourns the value of the visualizer
		self.visual1 = ifft(self.vort1ft, s=(self.V,self.V)).real
		self.visual2 = ifft(self.vort2ft, s=(self.V,self.V)).real

	def renoise(self):
	
		# We adjourn the noise
#		self.noise = 0.5*np.random.normal(loc = 0.0, scale =1.0, size
#								=(self.N, self.N))*np.sqrt(self.N*self.dt)
		self.noise =  np.sin(30*np.pi*self.X)*np.random.normal(loc = 0.0, scale = 1.0)*np.sqrt(self.dt)
	
	@timeit
	def implicit_euler(self):
	
		# We do one more step in the implicit Euler approximation

		# We start by computing the nonlinearity and the noise
		self.force()
		self.renoise()

		# This is the step forward in the Euler scheme
		self.vort1ft = np.multiply(self.relap, self.vort1ft +
							self.dt*(self.force1)+ fft(self.noise), dtype = complex)
		self.vort2ft = np.multiply(self.relap, self.vort2ft +
							self.dt*(self.force2)+ fft(self.noise), dtype = complex)
							
		# We adjourn both other values
		self.evaluate()
		self.visualize()

def animate(i):

	global vo, ax, fig, time_text

	# Real time is:
	ani_time = i*vo.dt

	# Set the new data
	im1.set_data(vo.vort1)
	im2.set_data(vo.vort2)
	

	# Set the new time
	time_text1.set_text("Time = {:2.3f}".format(ani_time))
	time_text2.set_text("Time = {:2.3f}".format(ani_time))

	# We print the step we are in
	sys.stdout.flush()
	sys.stdout.write("\r Step = {}".format(i))

	# And we do the next step:
	vo.implicit_euler()
	
	return [im1] + [im2] + [time_text1,] +[time_text2,]

# We initiate our solver
vo = vorticity()


from PIL import Image, ImageDraw

images = []

# We set up the picture
fig       = plt.figure(figsize = (19, 8))
plt.title("Vorticity with shear flow and low viscosity (0.1)")
plt.axis('off')

# And the two subplots
ax1       = fig.add_subplot(1,2,1)
ax2       = fig.add_subplot(1,2,2)

# Add axis limits
ax1.set_xlim(0, vo.N)
ax1.set_ylim(0, vo.N)
ax2.set_xlim(0, vo.N)
ax2.set_ylim(0, vo.N)

# But we do not plot the axis
ax1.axis('off')
ax2.axis('off')

# And time counter
time_text1 = ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax1.transAxes, color = 'white')

time_text2 = ax2.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax2.transAxes, color = 'white')


# Picture for the two initial conditions
im1       = ax1.imshow(vo.vort1, cmap = plt.get_cmap('jet'), vmin = -1, vmax = 1)
im2       = ax2.imshow(vo.vort2, cmap = plt.get_cmap('jet'), vmin = -1, vmax = 1)
	

for i in range(0, 200):

	# Real time is:
	ani_time = i*vo.dt

	# Set the new data
	im1.set_data(vo.vort1)
	im2.set_data(vo.vort2)
	
	# Set the new time
	time_text1.set_text("Time = {:2.3f}".format(ani_time))
	time_text2.set_text("Time = {:2.3f}".format(ani_time))

	plt.savefig('vorticity_fig_lv'+str(i)+'.png')
	
	for k in range(0, 28):
		vo.implicit_euler()
		print(i,k)

## We set up the picture
#fig       = plt.figure(figsize = (40, 20))
#plt.title("Vorticity equation")
#plt.axis('off')
#
## And the two subplots
#ax1       = fig.add_subplot(1,2,1)
#ax2       = fig.add_subplot(1,2,2)
#
## Add axis limits
#ax1.set_xlim(0, vo.N)
#ax1.set_ylim(0, vo.N)
#ax2.set_xlim(0, vo.N)
#ax2.set_ylim(0, vo.N)
#
## But we do not plot the axis
#ax1.axis('off')
#ax2.axis('off')
#
## And time counter
#time_text1 = ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax1.transAxes, color = 'white')
#
#time_text2 = ax2.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax2.transAxes, color = 'white')
#
## Picture for the two initial conditions
#im1       = ax1.imshow(vo.vort1, cmap = plt.get_cmap('rainbow'), vmin = -1.3, vmax = 1.3)
#im2       = ax2.imshow(vo.vort2, cmap = plt.get_cmap('rainbow'), vmin = -1.3, vmax = 1.3)
#
#
### We let the animation run.
##ani = FuncAnimation(fig, animate, frames= 2, repeat=False)
##mywriter = animation.PillowWriter(fps=30,bitrate=6000000)
##ani.save('vorticity.gif',writer=mywriter)
