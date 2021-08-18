# MCF via the Bence - Merriam - Osher algorithm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from matplotlib import colors
from IPython import embed
from os import path
from scipy import signal





def HeatKernel(t,x):
	# 2D Heat kernel
	return np.exp(-np.linalg.norm(x)**2/(4*t))/(4*np.pi*t)


class MCF:
	"""
	Here we represent the mean curvature flow
	"""
	def __init__(self, space_pts, LL, dt):

		# Initialize parameters
		self.space_pts = space_pts
		self.LL = LL
		self.space = np.linspace(0.0, LL, space_pts)
		self.space_fl = float(self.space_pts)
		self.dt = dt

		# State of the system:
		self.state = np.zeros(shape = (space_pts, space_pts))

		# Heat kernel at time 0.5
		self.hk = np.zeros(shape = (space_pts, space_pts))
		for i in range(0, space_pts):
			for j in range(0, space_pts):
				self.hk[i,j] = HeatKernel(0.5, self.Pos(i,j))/np.sqrt(self.space_fl/self.LL)

		# Heat kernel at time dt
		# We normalize it so that the convolution is an approximated Riemman integral.
		self.hk_dt = np.zeros(shape = (space_pts, space_pts))
		for i in range(0, space_pts):
			for j in range(0, space_pts):
				self.hk_dt[i,j] = HeatKernel(self.dt, self.Pos(i,j)+0.08)*self.LL/self.space_fl

		# We add the padding of the heatkernel, for the convolution
		#self.hk_dt_pad = np.pad(self.hk_dt,  ((space_pts,space_pts),(space_pts,space_pts))) 

		# The first step is to sample the correct Gaussian field.
		# This is not scaled like white noise (the scaling is implicit in the convolution)
		self.state = np.random.normal(loc=0.0, scale = 1.0, size = (space_pts, space_pts))
		self.state = signal.fftconvolve(self.state, self.hk, mode ='same')
		self.state = np.sign(self.state)


	def Pos(self, i, j):

		return np.array( [float(i)*self.LL/self.space_fl-self.LL/2, float(j)*self.LL/self.space_fl-self.LL/2])

	def do_step(self):

		self.state = signal.fftconvolve(self.state, self.hk_dt, mode='same')
		self.state = np.sign(self.state)


def animate(i):
	
	# Real time is:
	ani_time = i*dt
	# Redefine the plot
	im.set_data(MyMCF.state)
	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	# We print the step we are in:
	print(i)
	# And we do the next step:
	MyMCF.do_step()

	return [im,] + [time_text,]

# Time discretization
dt = 0.03

# Space discretisation
space_pts = 2000
LL = 150.0
space = np.linspace(0.0, LL, space_pts)

# This is our MCF
MyMCF = MCF(space_pts, LL, dt)
init  = MyMCF.state

#MyMCF.do_step


# We set up the picture
fig       = plt.figure()
ax        = plt.axes(xlim=(60, space_pts-60), ylim = (150, space_pts-150))
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
im        = ax.imshow(MyMCF.state, interpolation='none', vmin = -1.2, vmax = 1.2, cmap ='bwr')
#colmap    = plt.get_cmap('plasma') OR 'hot'
plt.title("MCF with Gaussian I.C.")

# We let the animation go.
ani = FuncAnimation(fig, animate, frames= 300, repeat=False)
mywriter = animation.FFMpegWriter(fps=10, codec="libx264", bitrate=60000, extra_args=['-pix_fmt', 'yuv420p'])
ani.save('mcf.mp4',writer=mywriter)




