
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

""" We compute the Lyapunov exponent of a product of random matrices that approximate the Anderson model """

class anderson():

	""" We create a random Hamiltonian according to the law that we want """
	
	def __init__(self, tau):
		super(anderson, self).__init__()

		# Parameters of the model
		# Time after which we renew the potential
		self.tau = tau 

		# Space-Time discretisation
		self.delta_t = 1/100
		self.delta_x = 1/180

		# Box size:
		L = 1.0

		# Space discretisation
		self.space = np.arange(0.0, 2*np.pi*L + 0.001, self.delta_x)
		self.space_pts = len(self.space)

		# Initial condition
		self.state = np.ones(shape = self.space_pts)

		# Count if we have to renew the noise
		self.count = 0

		# This is the resolvent of the laplacian matrix:
		# It is the periodic laplacian, and we normalize
		# the matrix (1-Delta) to have 1 on the diagonal.
		self.off_value = 0.5*(1/(1+2*(self.delta_t/self.delta_x**2)) -1)
		self.main_diag = np.ones(shape = (self.space_pts))
		self.offu_diag = self.off_value*np.ones(shape = (self.space_pts-1))
		self.to_invert = scipy.sparse.diags([self.offu_diag, self.main_diag, self.offu_diag], [-1, 0, 1]).toarray()

		#This line makes the resolvent periodic.
		self.to_invert[0,self.space_pts-1] = self.off_value
		self.to_invert[self.space_pts-1,0] = self.off_value

		# We then invert the matrix to get the resolvent.
		self.resolvent = scipy.linalg.inv(self.to_invert)/(1+2*(self.delta_t/self.delta_x**2))

		self.noise = np.zeros(shape=self.space_pts)
		self.renew_noise()

	def renew_noise(self):

		# This is for space white noise
		self.noise = 10*np.random.normal(0, np.sqrt(self.space_pts), self.space_pts)
		

	def do_step(self):

		while self.count*self.delta_t <= self.tau:
			
			# We do one iteration with the resolvent
			self.state = np.dot(self.resolvent, self.state + self.delta_t*(np.multiply(self.state, self.noise)))

			# We adjourn the counter
			self.count += 1

		# And we renew the noise
		self.renew_noise()
		self.count = 0

def normalize(vect_input):

	return vect_input/np.sum(vect_input)	

def angle(vect_input):

	return np.arccos(vect_input[1])
		

# We define the set of parameters we want to work with:
tau = np.arange(0.01, 0.9,0.05)
NN = len(tau)
TRYALS = 6
results = np.zeros(shape = (NN,TRYALS))


for y in range(NN):
	
	for z in range(TRYALS):
		
		AA = anderson(tau[y])
		AA.renew_noise()


		S = 0
		Snew = 0
		DeltaS =1
		x = 0

		eps = 1e-4
		count = 0 

		while (DeltaS >= eps or count <=10):
			
			AA.do_step()

			Snew = S*(1 - 1.0/(float(x)+1)) + np.log(np.sum(AA.state))/(float(x)+1)

			DeltaS = np.abs(S - Snew)
			S = Snew

			AA.state = normalize(AA.state)

			#print("\n ")
			# sys.stdout.flush()
			# sys.stdout.flush()
			# sys.stdout.flush()
			sys.stdout.write(" \nCurrent estimate {:2.8f}, with error {:2.8f}. \nWe are in (S, y, z, NN, tau) = ({}, {}, {}, {}, {}, {})".format(S, DeltaS, results[y,np.max(z-1,0)], y, z, NN, tau[y], count))
			#print("\n")
			x +=1
			if DeltaS < eps:
				count +=1
#		print(S, y, z, NN, tau[y])

		results[y,z] = S


fig, axs = plt.subplots(2)
avrg_results = results.mean(1)

axs[0].scatter(tau, avrg_results/tau)
axs[1].scatter(tau, avrg_results)

plt.show()

