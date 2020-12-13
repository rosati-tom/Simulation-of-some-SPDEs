
# Solve the KPZ equation for large times
# And two different initial conditions
# We use Space-Time discretesation (1D Finite elements)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from matplotlib import colors
from IPython import embed
from os import path
import sys

class she:
	"""
	We model the multiplicative SHE (1D).
	"""
	def __init__(self, x_0, x_1):
		# Initial state of the first system.
		self.state_a = x_0

		# Initial state of the second system.
		self.state_b = x_1

		# We initialize the noise variable.
		self.noise =  np.random.normal(size = (space_pts) , scale = np.sqrt(delta_t/delta_x))


	def do_step(self):
		# We renew the noise:
		self.noise =  np.random.normal(size = (space_pts) , scale = np.sqrt(delta_t/delta_x))

		# We do one more step in the implicit Euler approximations
		self.state_a = np.dot(resolvent, self.state_a - (np.multiply(np.multiply(self.state_a, self.state_a-1), self.state_a-0.5) )*delta_t + self.noise )

		self.state_b = np.dot(resolvent, self.state_b - (np.multiply(np.multiply(self.state_b, self.state_b-1), self.state_b-0.5) )*delta_t + self.noise )
		#self.state = np.dot(resolvent, self.state + np.random.normal(size = (space_pts), scale = np.sqrt(delta_t/delta_x) ) ) 

def animate(i):
	# global she_sample, ax, fig, time_text
	# Real time is:
	ani_time = i*delta_t

	# Redefine the plot
	lines_a.set_data(space, np.log(she_sample.state_a))
	lines_b.set_data(space, np.log(she_sample.state_b))

	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	# We print the step we are in:
	sys.stdout.flush()
	sys.stdout.write("\r Step = {}".format(i))
	# And we do the next step:
	she_sample.do_step()
	return [lines_a,] + [lines_b,] + [time_text,]

# Time discretisation
delta_t = 1/600
delta_x = 1/900

# Space discretisation
space = np.arange(0.0, 2*np.pi + 0.001, delta_x)
space_pts = len(space)

# We create a sample path
# with initial condition x_0, x_1:
x_0 = 1+0.5*np.sin(2*space)
x_1 = 1+0.2*space
she_sample = she(np.exp(x_0), np.exp(x_1))

# This is the resolvent of the laplacian matrix:
# It is the periodic laplacian, and we normalize
# the matrix (1-Delta) to have 1 on the diagonal.
off_value = 0.5*(1/(1+2*(delta_t/delta_x**2)) -1)
main_diag = np.ones(shape = (space_pts))
offu_diag = off_value*np.ones(shape = (space_pts-1))
to_invert = scipy.sparse.diags([offu_diag, main_diag, offu_diag], [-1, 0, 1]).toarray()

#This line makes the resolvent periodic.
to_invert[0,space_pts-1] = off_value
to_invert[space_pts-1,0] = off_value

#We then invert the matrix to get the resolvent.
resolvent = scipy.linalg.inv(to_invert)/(1+2*(delta_t/delta_x**2))

#We set up the picture
fig       = plt.figure()
ax        = plt.axes(xlim=(0, np.pi), ylim = (-1, 2.0))
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
lines_a,  = ax.plot([],[], lw = 2)
lines_b,  = ax.plot([],[], lw = 2)
plt.title("Synchronization") 

# We let the animation go.
ani       = animation.FuncAnimation(fig, animate, frames=250, interval = 70, blit = True)
ani.save(filename = 'kpz_synchronization.mp4', extra_args=['-vcodec', 'libx264'], bitrate = 17000)


# INSTRUCTION FOR PUTTING VIDEO IN PRESENTATION.

# 1) RUN: ffmpeg -i <input> -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 25 -r 25 output.mp4
#	 on powershell. The result (output.mp4) is the video you will use.
# 2)  IN Latex, with package movie9 write:
#   \includemedia[
#  width=0.7\linewidth,
#  totalheight=0.7\linewidth,
#  activate=onclick,
#  %passcontext,  %show VPlayer's right-click menu
#  addresource=ballistic_out.mp4,
#  flashvars={
#    %important: same path as in `addresource'
#    source=ballistic_out.mp4
#  }
#]{\fbox{Click!}}{VPlayer.swf}