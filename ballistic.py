
# We model ballistic growth

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from IPython import embed
from os import path
import sys

class ballistic:
	"""
	We model ballistic aggregation.
	"""
	def __init__(self, M):
		# Initial state of the sistem.
		self.size = M
		# Time of next arrival at a given point.
		self.times = np.random.exponential(scale = 1.0, size = [M])
		# Next particle to move.
		self.move = np.argsort(self.times)[0]
		# We add the information of the system.
		# Namely: the current height:
		self.height   = np.zeros(shape = [M,2])
		self.arrival  = np.zeros(shape = [3,1])
		self.step     = 0
		# We save the current time (initial = first arrival time):
		self.cur_time = self.times[self.move]
		self.max_height = 0

	def do_step(self):
		# We let one particle arrive anc change the height.
		tm = self.move
		max_height = np.argsort( -np.array([self.height[(tm-1)%M, 0], self.height[(tm+1)%M, 0], (self.height[tm, 0]+1)])  )[0]
		if (max_height == 0):
			self.height[tm, 0] = self.height[(tm-1)%M, 0] 
		if (max_height == 1):
			self.height[tm, 0] = self.height[(tm+1)%M, 0]
		if (max_height == 2):
			self.height[tm, 0] += 1
		self.height[tm, 1] += self.times[tm]
		if (self.max_height < self.height[tm,1]):
			self.max_height = self.height[tm,1]
		# We add the arrival.
		st = self.step
		self.arrival[0,st] = tm
		self.arrival[1,st] = self.height[tm,0]
		self.arrival[2,st] = self.height[tm,1]
		
		# Finally, we add a new time:
		self.times[tm] += np.random.exponential(scale = 1.0, size = [1])
		self.move = np.argsort(self.times)[0]

		# And we add the step and the time of the next arrival:
		self.step    += 1
		self.cur_time = self.arrival[2, st]

		# And we add a new column to the arrivals.
		new_arrival  = np.zeros(shape = [3,1])
		self.arrival = np.append(self.arrival, new_arrival, axis=1)
		
def animate(i):
	global bal, ax, fig, time_text
	# Real time is:
	ani_time = (i/240)**3
	# We do as many steps as needed for the next particle
	# to arrive after time ani_time.
	while (ani_time >  bal.cur_time ):
		bal.do_step()
	n = bal.step
	scat.set_offsets(bal.arrival[0:2,:-1].transpose())

	# We color the data:
	norm = colors.Normalize(vmin=0.,vmax=bal.cur_time)
	col  = colmap(norm(bal.arrival[2, :-1]))
	#pcm = ax.pcolor(col, vmin=0., vmax=bal.cur_time, cmap='autumn')
	#col = ['#ff8000' for _ in range(0,n-1)]
	scat.set_color(col)

	# We print the step we are in:
	sys.stdout.flush()
	sys.stdout.write("\r Step = {}".format(i))

	# We zoom out and add time:
	if ( (0.001*ani_time)**2 > 1/60 ):
		scat.set_sizes([1/(0.001*ani_time)**2])
	ax.set_xlim(500-12.5-(0.08)*ani_time,500+12.5+(0.08)*ani_time)
	ax.set_ylim(0.01*ani_time, 0.09*ani_time + 25)
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	return [scat,] + [time_text,]

# How many particles
M   = 1000

# We create an example
bal = ballistic(M)

# We set up the picture
fig       = plt.figure()
ax        = plt.axes(xlim=(500-12.5, 500+12.5), ylim = (0, 25))
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
scat      = ax.scatter([], [], s=60, cmap = "plasma")
colmap    = plt.get_cmap('plasma')

# No numbers showing along the axes.
ax.set_xticks([])
ax.set_yticks([])
# We let the animation go.
ani       = animation.FuncAnimation(fig, animate, frames=8000,interval = 1, blit=True)
ani.save(filename = 'ballistic.mp4', fps=150, extra_args=['-vcodec', 'libx264'], bitrate = 20000)


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
