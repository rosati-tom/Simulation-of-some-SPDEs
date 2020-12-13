
# SHE MOVIE in 2D with heat map.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from IPython import embed
from os import path


class she:
	"""
	We model the SHE (2D).
	"""
	def __init__(self, x0):
		# Initial state of the sistem in terms of Fourier coefficients:
		self.state = np.zeros(shape = (N,N))
		self.state = x0
		# Initial state of the sistem in real coordinates:
		self.value = np.zeros(shape = (space_pts, space_pts))
		self.value = np.tensordot(eigenfct_tensor, self.state, axes = ([1,3],[0,1]))

	def do_step(self):
		# We do one more step in the implicit Euler approximations
		self.state = np.multiply(dd, self.state + np.random.normal(loc = 0.0, scale = np.sqrt(delta_t), size = (N,N)))
		self.value = np.zeros(shape = (space_pts, space_pts))
		self.value += np.tensordot(eigenfct_tensor, self.state, axes = ([1,3],[0,1]))
		
def animate(i):
	global she_sample, ax, fig, time_text
	# Real time is:
	ani_time = i*delta_t
	# Redefine the plot
	im.set_data(she_sample.value)
	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	# We print the step we are in:
	print(i)
	# And we do the next step:
	she_sample.do_step()
	return [im] + [time_text]

# Number of eigenfuntions
N = 40

# Time discretisation
delta_t = 1/1000

# Diagonal for implicit Euler.
dd = np.zeros(shape = (N, N))
for i in range(0, N):
	for j in range(0, N):
		dd[i, j] = 1/(1+((i+1)**2+ (j+1)**2)*delta_t)

# Space discretisation
space_pts = 300
space = np.linspace(0.0, np.pi + 1/(space_pts-10), space_pts)

# Define the eigenfunctions in 1D:
eigenfct_matrix = np.zeros(shape = (space_pts,N))
for i in range(0,N):
	eigenfct_matrix[:,i] = np.sin(i*space)*(np.sqrt(2/np.pi))

# Take tensor product in 2D:
# Gives a (space, N, space, N) tensor
eigenfct_tensor = np.tensordot(eigenfct_matrix, eigenfct_matrix, axes = 0)

# We create a sample path
# with initial condition x0:
x0 = np.zeros(shape = (N, N))
she_sample = she(x0)

# We set up the picture
fig        = plt.figure()
ax         = plt.axes(xlim=(0, space_pts), ylim = (0, space_pts))
time_text  = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
im         = plt.imshow(she_sample.value, vmin = -1, vmax = 1)
#colmap    = plt.get_cmap('plasma') OR 'hot'

# We let the animation go.
ani        = animation.FuncAnimation(fig, animate, frames=250,interval = 2, blit=True)
ani.save(filename = '2d_she.mp4', fps=15, extra_args=['-vcodec', 'libx264'], bitrate = 17000)

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