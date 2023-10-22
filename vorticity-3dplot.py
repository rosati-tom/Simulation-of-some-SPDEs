

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sys
from matplotlib import colors
from IPython import embed
from os import path
from mpl_toolkits import mplot3d
import time
from functools import wraps
import inspect

# We define inverese and direct fft reversely, because the renormalisation
# makes more sense: the fft should be an integral and thus have a factor 1/n in
# front of the noise.
from numpy.fft import fft2  as ifft
from numpy.fft import ifft2 as fft

# We also take the allegedly faster fft transforms from pyfftw
from pyfftw.interfaces.numpy_fft import fft2  as ifft_fast
from pyfftw.interfaces.numpy_fft import ifft2 as fft_fast
# INDEED: With fftw seems to be about 30% faster than the previous one.

# We rescale the axis to respect the x/y eps ratio
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

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

# This is the class containing the solver for the vorticity equation
class vorticity_fast:
	"""
	We simulate the vorticity equation with a shear force
	"""
	def __init__(self, viscosity):

		# Initial state of the system in terms of Fourier coefficients:
		# This is the number of eigenfunctions we use (for the definitions
		# below to work correctly with the Numpy implementation of the fast
		# Fourier transform this number should be odd: zeroth + first half positive,
		# + second half negative modes)
		self.N   = 2*round(1/(2*np.minimum(viscosity**2,0.03**2)))+1
		# used to be 700 at least
		# And we set the time discretization for our problem
		self.dt  = 0.01
		# And we set the parameter a for the viscosity
		self.nu  = viscosity
		
		# We define our space
		# This parameter sets the length of the y axis as self.exs^-1 (which we neeed to be long)
		self.eps = 1.6
		# Then we set up the space
		self.space_X = np.linspace(-0.5, 0.5, self.N)
		self.space_Y = np.linspace(-0.5/self.eps, 0.5/self.eps, self.N)
		self.X, self.Y = np.meshgrid(self.space_X, self.space_Y)
		
		# We define our two initial conditions
		self.vort1 = np.zeros(shape = (self.N,self.N), dtype = float)
		self.vort2 = np.zeros(shape = (self.N,self.N), dtype = float)
	
		self.vort1 = np.sin(2*np.pi*self.X)
		self.vort2 = np.sin(2*np.pi*self.eps*(self.Y+self.Y**2))
		
#		self.vort2 = np.sin(2*np.pi*self.Y+self.Y**2)+np.exp(self.X)
#		self.vort2 = np.multiply(self.relap, np.random.normal(loc = 0.0, scale =1.0, size =(self.N, self.N))*self.N)
		
		# And we define the noise, which we immediately update
		self.noise = np.zeros(shape = (self.N,self.N), dtype = float)
		self.renoise()

		# The initial condition in its Fourier coefficients
		self.vort1ft = fft_fast(self.vort1)
		self.vort2ft = fft_fast(self.vort2)
		
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
		
		# Placeholders for the force
		self.force1 = np.zeros(shape = (self.N,self.N))
		self.force2 = np.zeros(shape = (self.N,self.N))
		
		# Placeholders for the mean
		self.average1 = np.zeros(shape = self.N)
		self.average2 = np.zeros(shape = self.N)
		
		# Placeholders for the norm
		self.norm1 = np.zeros(shape = 1)
		self.norm2 = np.zeros(shape = 1)

		# We define the inverse Laplacian as a multiplier
		# This leaves the zeroth Fourier mode unchanged
		self.invlaplace = np.zeros(shape = (self.N, self.N), dtype= complex)
		for i in range(0,self.N):
			for k in range(0, self.N):
				if i>0 or k > 0:
					if i <self.N//2+1 and k <self.N//2+1:
						self.invlaplace[i,k] = complex(- 1/((self.eps**2)*float(i)**2 + float(k)**2),0.0)
					if i <self.N//2+1 and  k >= self.N//2+1:
						self.invlaplace[i,k] = complex(- 1/((self.eps**2)*float(i)**2 + (self.N-k)**2),0.0)
					if  i >= self.N//2+1 and  k < self.N//2+1:
						self.invlaplace[i,k] = complex(- 1/((self.eps**2)*float(self.N-i)**2 + float(k)**2),0.0)
					if i >= self.N//2+1 and  k >= self.N//2+1:
						self.invlaplace[i,k] = complex(- 1/((self.eps**2)*float(self.N-i)**2 + float(self.N-k)**2),0.0)
						
		# This is the multiplier for differentiation in the x variable
		self.nabla_y = np.zeros(shape = (self.N, self.N), dtype = complex)
		for i in range(0,self.N):
			for k in range(0, self.N):
				if i>0 or k > 0:
					if i <self.N//2+1 and k <self.N//2+1:
						self.nabla_y[i,k] = self.eps*complex(0.0, 1.0*i)
					if i <self.N//2+1 and  k >= self.N//2+1:
						self.nabla_y[i,k] = self.eps*complex(0.0, 1.0*i)
					if  i >= self.N//2+1 and  k < self.N//2+1:
						self.nabla_y[i,k] = self.eps*complex(0.0, -1.0*(self.N-i))
					if i >= self.N//2+1 and  k >= self.N//2+1:
						self.nabla_y[i,k] = self.eps*complex(0.0, -1.0*(self.N-i))
		
		# This is the multiplier for differentiation in the y variable
		self.nabla_x = np.zeros(shape = (self.N, self.N), dtype = complex)
		for i in range(0,self.N):
			for k in range(0, self.N):
				if i>0 or k > 0:
					if i <self.N//2+1 and k <self.N//2+1:
						self.nabla_x[i,k] = complex(0.0, 1.0*k)
					if i <self.N//2+1 and  k >= self.N//2+1:
						self.nabla_x[i,k] = complex(0.0, -1.0*(self.N-k))
					if  i >= self.N//2+1 and  k < self.N//2+1:
						self.nabla_x[i,k] = complex(0.0, 1.0*k)
					if i >= self.N//2+1 and  k >= self.N//2+1:
						self.nabla_x[i,k] = complex(0.0, -1.0*(self.N-k))
						
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
						self.relap[i,k] = complex(1.0/(1.0+(self.nu*(self.eps**2)*i**2+self.nu*k**2)*self.dt), 0.0)
					if i <self.N//2+1 and  k >= self.N//2+1:
						self.relap[i,k] = complex(1.0/(1.0+(self.nu*(self.eps**2)*i**2+self.nu*(self.N -k)**2)*self.dt), 0.0)
					if  i >= self.N//2+1 and  k < self.N//2+1:
						self.relap[i,k] = complex(1.0/(1.0+(self.nu*(self.eps**2)*(self.N-i)**2+ self.nu*k**2)*self.dt), 0.0)
					if i >= self.N//2+1 and  k >= self.N//2+1:
						self.relap[i,k] = complex(1.0/(1.0+(self.nu*(self.eps**2)*(self.N-i)**2+ self.nu*(self.N-k)**2)*self.dt), 0.0)
		
	def force(self):
		
		# We compute the gradient
		self.grad1_x = ifft_fast(np.multiply(self.nabla_x, self.vort1ft, dtype = complex)).real
		self.grad1_y = ifft_fast(np.multiply(self.nabla_y, self.vort1ft, dtype = complex)).real
		
		self.grad2_x = ifft_fast(np.multiply(self.nabla_x, self.vort2ft, dtype = complex)).real
		self.grad2_y = ifft_fast(np.multiply(self.nabla_y, self.vort2ft, dtype = complex)).real
		
		# We compute the stream function
		self.stream1_x = ifft_fast(np.multiply(self.invlaplace_x, self.vort1ft, dtype = complex)).real
		self.stream1_y = ifft_fast(np.multiply(self.invlaplace_y, self.vort1ft, dtype = complex)).real
		
		self.stream2_x = ifft_fast(np.multiply(self.invlaplace_x, self.vort2ft, dtype = complex)).real
		self.stream2_y = ifft_fast(np.multiply(self.invlaplace_y, self.vort2ft, dtype = complex)).real
		
		# We compute the force in Fourier coordinates
		self.force1 = fft_fast(-np.multiply(self.stream1_y, self.grad1_x, dtype = complex) + np.multiply(self.stream1_x, self.grad1_y, dtype = complex))
		
		self.force2 = fft_fast(-np.multiply(self.stream2_y, self.grad2_x, dtype = complex) + np.multiply(self.stream2_x, self.grad2_y, dtype = complex))
		
		
	def evaluate(self):
		# This function adjourns the value of the real state of the system.
		self.vort1 = ifft_fast(self.vort1ft, s = (self.N, self.N)).real
		self.vort2 = ifft_fast(self.vort2ft, s = (self.N, self.N)).real

	def renoise(self):
	
		# We adjourn the noise
		self.noise =  3*np.sqrt(self.nu)*np.sin(8*np.pi*self.X)*np.random.normal(loc = 0.0, scale = 1.0)*np.sqrt(self.dt)
	
#	@timeit
	def implicit_euler(self):
	
		# We do one more step in the implicit Euler approximation

		# We start by computing the nonlinearity and the noise
		self.renoise()

		# This is the step forward in the Euler scheme
		self.vort1ft = np.multiply(self.relap, self.vort1ft +
							self.dt*(self.force1)+ fft_fast(self.noise), dtype = complex)
		self.vort2ft = np.multiply(self.relap, self.vort2ft +
							self.dt*(self.force2)+ fft_fast(self.noise), dtype = complex)
							
		# We adjourn all other values
		self.force()
		self.evaluate()
		self.average()
		
	def average(self):
		
		# We compute the average over the X direction, so that we obtain a function that depends on Y
		# In particular, if the solution becomes shear this quantity should be constant
		
		self.average1 = np.mean(self.vort1, axis=1)
		self.average2 = np.mean(self.vort2, axis=1)
		
	def L2(self):
		
		# We compute the L2 norm of the average over the X direction
		
		self.norm1 = np.norm(self.average1)
		self.norm2 = np.norm(self.average2)

def animate(i):

	global vo_hv, vo_lv, ax, surf1, surf2, surf3, surf4

	# Real time is:
	ani_time = i*vo_hv.dt

	# Set the new data for the first subplot
	surf1.remove()
	surf1=ax1.plot_surface(vo_hv.X, vo_hv.Y, vo_hv.vort1, cmap='jet',
                       linewidth=0, antialiased=False)
	
	# Set the new data for the second subplot
	surf2.remove()
	surf2=ax2.plot_surface(vo_hv.X, vo_hv.Y, vo_hv.vort2, cmap='jet',
                       linewidth=0, antialiased=False)
                       
	# Set the new data for the third subplot
	surf3.remove()
	surf3=ax3.plot_surface(vo_lv.X, vo_lv.Y, vo_lv.vort1, cmap='jet',
                       linewidth=0, antialiased=False)

	# Set the new data for the fourth subplot
	surf4.remove()
	surf4=ax4.plot_surface(vo_lv.X, vo_lv.Y, vo_lv.vort2, cmap='jet',
                       linewidth=0, antialiased=False)
                       
	# Set data for the last two plots
	lines_a.set_data(vo_hv.space_Y, vo_hv.average2)
	lines_b.set_data(vo_lv.space_Y, vo_lv.average2)

	# Set the new time
	time_text.set_text("Length ratio = {:2.1f}. Time = {:2.2f}".format(vo_hv.eps, ani_time))

	# We print the step we are in
	sys.stdout.flush()
	sys.stdout.write("\r Step = {}".format(i))

	# And we do the next step:
	for i in range(0,1):
		vo_hv.implicit_euler()
		vo_lv.implicit_euler()
	
	return [surf1,] , [surf2,], [surf3,] , [surf4,], [lines_a,], [lines_b,]
	
# We initiate our solver

# First one with high viscosity
vo_hv = vorticity_fast(2.00)

# Second one with low viscosity
# To compare to the previous one, we here use fast
vo_lv = vorticity_fast(1.00)

# We set up the plot
fig = plt.figure(figsize=(40,20), dpi=20)

# Top three are high viscosity
ax1       = fig.add_subplot(2,3,1, projection='3d')
ax2       = fig.add_subplot(2,3,2, projection='3d')
ax5       = fig.add_subplot(2,3,3)

ax3       = fig.add_subplot(2,3,4, projection='3d')
ax4       = fig.add_subplot(2,3,5, projection='3d')

ax6       = fig.add_subplot(2,3,6)

#
## And for the energy
#ax7       = fig.add_subplot(4,2,7)
#ax8       = fig.add_subplot(4,2,8)

# Set up axes an stuff for the 2D plot
ax5.set_xlim(-0.5/vo_lv.eps, 0.5/vo_lv.eps)
ax5.set_ylim(-1.4, 1.4)
ax5.set_axis_off()

ax6.set_xlim(-0.5/vo_lv.eps, 0.5/vo_lv.eps)
ax6.set_ylim(-1.4, 1.4)
ax6.set_axis_off()


lines_a,  = ax5.plot([],[], lw = 2)
lines_b,  = ax6.plot([],[], lw = 2)

# We st up the 3D plot

# Set labels
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Vorticity');
# We set the axis off for aesthetic reasons
ax1.set_axis_off()

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Vorticity');
ax2.set_axis_off()

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Vorticity');
ax3.set_axis_off()

ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('Vorticity');
ax4.set_axis_off()

# Define surface plots
surf1 = ax1.plot_surface(vo_hv.X, vo_hv.Y, vo_hv.vort1, cmap='jet',
                       linewidth=0, antialiased=False)
surf2 = ax2.plot_surface(vo_hv.X, vo_hv.Y, vo_hv.vort2, cmap='jet',
                       linewidth=0, antialiased=False)
surf3 = ax3.plot_surface(vo_lv.X, vo_lv.Y, vo_lv.vort1, cmap='jet',
                       linewidth=0, antialiased=False)
surf4 = ax4.plot_surface(vo_lv.X, vo_lv.Y, vo_lv.vort2, cmap='jet',
                       linewidth=0, antialiased=False)

# Set z axis limits
ax1.set_zlim(-2.01, 2.01)
ax2.set_zlim(-2.01, 2.01)
ax3.set_zlim(-2.01, 2.01)
ax4.set_zlim(-2.01, 2.01)

# We scale the axis, so that the x/y ration is preserved
x_scale=1.0
y_scale=1/vo_lv.eps
z_scale=1.0

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(2.6/scale.max())
scale[3,3]=1.0

ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), scale)
ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), scale)
ax3.get_proj = lambda: np.dot(Axes3D.get_proj(ax3), scale)
ax4.get_proj = lambda: np.dot(Axes3D.get_proj(ax4), scale)

# And texts
time_text = ax1.text(7.05, 2.95, 69.95,'',horizontalalignment='left',verticalalignment='bottom', transform=ax1.transAxes, color = 'black', size=30)
text1 = ax1.text(0.5, 2.95, 62.95,"Shear initial condition \nHigh viscosity: nu = {:2.3f}".format(vo_hv.nu),horizontalalignment='left',verticalalignment='bottom', transform=ax1.transAxes, color = 'black', size=30)
text2 = ax2.text(0.5, 2.95, 62.95,"Non-shear initial condition, \nHigh viscosity: nu = {:2.3f}".format(vo_hv.nu),horizontalalignment='left',verticalalignment='bottom', transform=ax2.transAxes, color = 'black', size=30)
text3 = ax3.text(0.5, 2.95, -27.95,"Shear initial condition \nLow viscosity: nu = {:2.3f}".format(vo_lv.nu),horizontalalignment='left',verticalalignment='bottom', transform=ax1.transAxes, color = 'black', size=30)
text4 = ax4.text(0.5, 2.95, -27.95,"Non-shear initial condition \nLow viscosity: nu = {:2.3f}".format(vo_lv.nu),horizontalalignment='left',verticalalignment='bottom', transform=ax2.transAxes, color = 'black', size=30)
#text5 = ax2.text(20.5, 2.95, 62.95,"Non-shear initial condition profile \nHigh viscosity: nu = {:2.1f}".format(vo_hv.nu),horizontalalignment='left',verticalalignment='bottom', transform=ax2.transAxes, color = 'black', size=30)
text6 = ax2.text(20.5, 5.95, 2.95,"Non-shear initial condition profiles",horizontalalignment='left',verticalalignment='bottom', transform=ax2.transAxes, color = 'black', size=30)


# We let the animation run.
ani = FuncAnimation(fig, animate, frames=200, repeat=False)
mywriter = animation.PillowWriter(fps=16,bitrate=600000000)
ani.save('vorticity-3d-try.gif',writer=mywriter, dpi = 100)
