
# We simulate the Brownian sheet on [0, T] x [0, pi]


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from IPython import embed
from os import path


class b_sheet:
	"""
	We model the multiplicative SHE (1D).
	"""
	def __init__(self):
		# Initial state of the sistem.
		self.cur_state = np.zeros(shape = (space_pts))

	def do_step(self):
		self.cur_state += np.cumsum(np.random.normal(size = (space_pts), scale = np.sqrt(delta_t*delta_x)))

def animate(i):
	global b_expl, ax, fig, time_text
	# Real time is:
	ani_time = i*delta_t
	# Redefine the plot
	lines.set_data(space, b_expl.cur_state)
	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	# We print the step we are in:
	print(i)
	# And we do the next step:
	b_expl.do_step()
	return [lines,] + [time_text,]

# Time discretisation
delta_t = 1/1000

# Space discretisation
delta_x = 0.001
space = np.arange(0.0, np.pi + 0.001, delta_x)
space_pts = len(space)

# We create a sample path
b_expl = b_sheet()

# We set up the picture
fig       = plt.figure()
ax        = plt.axes(xlim=(0, np.pi), ylim = (-3, 3))
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
lines,    = ax.plot([],[], lw = 1)

# We let the animation go.
ani       = animation.FuncAnimation(fig, animate, frames=700,interval = 1, blit=True)
ani.save(filename = 'b_sheet.mp4', fps=20, extra_args=['-vcodec', 'libx264'], bitrate = 3000)


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