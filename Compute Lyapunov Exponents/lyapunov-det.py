

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


""" We plot the pathwhise total mass of random matrix. """


class r_matrix():

	"""We create a random matrix according to the law that we want"""
	
	def __init__(self, tau, nu):
		super(r_matrix, self).__init__()

		# Parameters of the model
		# Time after which we renew the potential
		self.tau = tau
		# Viscosity parameter
		self.nu = nu
		self.base_matrix = np.zeros((3,3))+self.nu
		self.exp_mat =sp.linalg.expm(self.tau*self.base_matrix)

		# This is if we want to add a dependence in time
		self.count = 0


	def sample(self):

		""" Different random matrices to choose from """


		# # 3x3 matrix case with biased noise
		# self.prob= 0.1
		# self.base_matrix[0,0]= - 2*self.nu + (np.random.binomial(n=1,p=self.prob))/self.prob-1
		# self.base_matrix[1,1]= - 2*self.nu + (np.random.binomial(n=1,p=self.prob))/self.prob-1
		# self.base_matrix[2,2]= - 2*self.nu + (np.random.binomial(n=1,p=self.prob))/self.prob-1

		# 3x3 matrix case
		self.base_matrix[0,0]= - 2*self.nu + np.random.normal(loc=0.0, scale=2.0, size=None)
		self.base_matrix[1,1]= - 2*self.nu + np.random.normal(loc=0.0, scale=2.0, size=None)
		self.base_matrix[2,2]= - 2*self.nu + np.random.normal(loc=0.0, scale=2.0, size=None)

		self.exp_mat =sp.linalg.expm(self.tau*self.base_matrix)
		# self.count = (self.count+1)%2

	def apply(self, vect_input):

		return self.exp_mat.dot(vect_input)

		
# We define the set of parameters we want to work with:
tau        = 0.002
time_steps = np.arange(0.01, 1.2, tau)
time_len   = len(time_steps)

results         = np.zeros(shape = time_len)
results_der     = np.zeros(shape = time_len)
results_sec_der = np.zeros(shape = time_len)

MM = r_matrix(tau, 0.2)

MM.sample()

VV = np.zeros(shape = 3) + 1.0

fig, axs = plt.subplots(3)

TRYALS = 10

for a in range(TRYALS):

	for i in range(time_len):

		VV = MM.apply(VV)

		results[i] = np.average(VV)
		if i>=1:
			results_der[i] = (results[i]-results[i-1])/tau

		if i>=2:
			results_sec_der[i] = (results[i] - 2*results[i-1] + results[i-2])/(tau)**2

	#axs[0].plot(time_steps, (results-results[0])/time_steps)

	axs[0].plot(time_steps, results)

	axs[1].plot(time_steps, results_der)

	axs[2].plot(time_steps, results_sec_der)

	# We resample the initial condition at random

	VV = np.random.uniform(low = 0.01, high = 3.01, size = 3)
	VV = VV / np.average(VV)

plt.show()










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
