

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


""" We compute the Lyapunov exponent of a product of random matrices """


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


	def renew(self):


		# # 3x3 matrix case with biased noise
		# self.prob= 0.1
		# self.base_matrix[0,0]= - 2*self.nu + (np.random.binomial(n=1,p=self.prob))/self.prob-1
		# self.base_matrix[1,1]= - 2*self.nu + (np.random.binomial(n=1,p=self.prob))/self.prob-1
		# self.base_matrix[2,2]= - 2*self.nu + (np.random.binomial(n=1,p=self.prob))/self.prob-1

		# 3x3 matrix case
		self.base_matrix[0,0]= - 2*self.nu + np.random.normal(loc=0.0, scale=1.0, size=None)
		self.base_matrix[1,1]= - 2*self.nu + np.random.normal(loc=0.0, scale=1.0, size=None)
		self.base_matrix[2,2]= - 2*self.nu + np.random.normal(loc=0.0, scale=1.0, size=None)


		# if self.count ==0 :
		
		# 	self.base_matrix[0,0]= -self.nu + np.random.normal(loc=0.0, scale=1.0, size=None)
		# 	self.base_matrix[1,0]= self.nu
		# 	self.base_matrix[0,1]= self.nu
		# 	self.base_matrix[1,1]= - self.nu + np.random.normal(loc=0.0, scale=1.0, size=None)

		# if self.count ==1 :
			
		# 	self.base_matrix[0,0]= -self.nu - (self.base_matrix[0,0] + self.nu)
		# 	self.base_matrix[1,0]= self.nu
		# 	self.base_matrix[0,1]= self.nu
		# 	self.base_matrix[1,1]= -self.nu - (self.base_matrix[1,1] + self.nu)


		self.exp_mat =sp.linalg.expm(self.tau*self.base_matrix)
		# self.count = (self.count+1)%2

	def apply(self, vect_input):

		return self.exp_mat.dot(vect_input)


class measure():
	"""This keeps track of the density of the projective invariant measure"""

	def __init__(self, dx, x_init, x_fin):
		super(measure, self).__init__()
		
		self.dx = dx
		self.x_init = x_init
		self.x_fin = x_fin
		self.sample_size = 0
		
		# We create the bins
		self.space_points = np.arange(self.x_init, self.x_fin, self.dx)
		self.len = len(self.space_points)

		self.normalized_bins = np.zeros(shape= self.len)

	def add(self, data_point):

		self.data_point = data_point

		# We compute the right bin
		self.bin_num = next(x for x in range(self.len) if self.space_points[x+1] > self.data_point)

		# We add the data point to the right bin and change all other variables accordingly
		self.normalized_bins = self.normalized_bins*(self.sample_size/(self.sample_size+1)) 
		self.normalized_bins[self.bin_num] += self.data_point*(self.sample_size/(self.sample_size+1))*self.dx
		
		self.sample_size +=1

	def clear(self):

		self.normalized_bins = np.zeros(shape= self.len)
		self.sample_size =0


def normalize(vect_input):

	return vect_input/np.linalg.norm(vect_input)	


def angle(vect_input):

	return np.arccos(vect_input[1])

		
# We define the set of parameters we want to work with:
tau = np.arange(0.03,3.1,0.02)

NN = len(tau)

TRYALS = 12

results = np.zeros(shape = (NN,TRYALS))

# Proj_measure = measure(0.03, 0.0, np.pi/2.0)
# len_measure = Proj_measure.len
# results_measure = np.zeros(shape = (NN, TRYALS, len_measure))

for y in range(NN):
	
	for z in range(TRYALS):
		
		AA = r_matrix(tau[y], 1.0)
		AA.renew()

		# Proj_measure.clear()

		DIR = np.array([1,1,1]) 

		LIM = 7000

		S = 0


		for x in range(LIM):

			NEXT = AA.apply(DIR)

			S = S*(float(x)/(float(x)+1)) + np.log(np.linalg.norm(NEXT))/(float(x)+1)

			DIR = normalize(NEXT)
			AA.renew()

			# Proj_measure.add(angle(DIR))

		print(S, y, z, NN, tau[y])

		results[y,z] = S
		# results_measure[y,z,:] = Proj_measure.normalized_bins


fig, axs = plt.subplots(2)

avrg_results = results.mean(1)
# avrg_results_measure =results_measure.mean(1)

axs[0].scatter(tau, avrg_results/tau)

axs[1].scatter(tau, avrg_results)

# for x in range(NN):

	# axs[1].plot(Proj_measure.space_points, avrg_results_measure[x,:])

plt.show()

embed()










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
