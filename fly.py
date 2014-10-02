import numpy as np
from scipy.interpolate import UnivariateSpline
from readSimion import EField3D, EField2D
from scipy.io import loadmat

from time import time

from matplotlib import pyplot as plt

from generateWaveformPotentials import generateWaveformPotentials

# CONSTANTS
# mass of hydrogen
mass = 1.6735327160314e-27
# Boltzmann constant / J/K
kB = 1.380648813e-23
# Bohr radius / m
a0 = 5.291772109217e-11
# elementary charge / C
e = 1.60217656535e-19

class rydberg_flyer(object):
	def __init__(self):
		pass
	
	def setInitialDistribution2D(self, filename, xWidth):
		A = loadmat(filename)
		
		# this is still a 3D distribution, so for the 2D simulation we're interested in here
		# we select a central slice only, and disregard the x-axis afterwards
		mask = (A['rLasersIntercept'][0] > -xWidth/2) & (A['rLasersIntercept'][0] < xWidth/2)
		self.pos = np.flipud(A['rLasersIntercept'][1:3, mask])
		self.vel = np.flipud(A['vLasersIntercept'][1:3, mask])
		
		# shift everything by rydberg excitation point
		rydbergExcitationPoint = np.array([[102, 87.5]])*np.array([[self.ef.dx, self.ef.dr]])*1E-3
		self.pos += rydbergExcitationPoint.T
		
	def load_fields(self, folder, scale, nElectrodes):
		self.ef = EField2D(folder, [0]*nElectrodes, scale)
		
		self.acc = ctypes.cdll.LoadLibrary('./' + target + '.so')
		self.acc.set_npas.argtypes = [c_uint]
		self.acc.set_npas.restype = None
		self.acc.add_pa.argtypes = [c_uint, c_double_p, c_double]
		self.acc.add_pa.restype = None
		self.acc.set_pasize.argtypes = [c_uint, c_uint, c_double, c_double]
		self.acc.set_pasize.restype = None
		self.acc.getFieldGradient.argtypes = [c_uint, c_double_p, c_double_p, c_double_p]
		self.acc.getFieldGradient.restype = None

		
		self.acc.set_npas(nElectrodes)
		self.acc.set_pasize(self.ef.nx, self.ef.ny, self.ef.dx, self.ef.dr)
		for n in range(nElectrodes):
			self.acc.add_pa(n, self.ef.pas[n].potential.ctypes.data_as(c_double_p), 0)

	
	def collision(self, pos, ef):
		collision_index = np.ones_like(pos)
		for f in ef:
			index = f.isElectrode(pos)
			collision_index[index, :] = 0
		
		return collision_index
	
	@profile
	def fly_atoms(self, potentialPCB, n, deltaT):
		k = n - 1
		
		posCurrent = np.zeros_like(self.pos)
		rxCurrent = posCurrent[:, 0]
		ryCurrent = posCurrent[:, 1]
		
		# split solutions into vectors for positions r_ and velocity v_
		# positon includes time steps t_-2=rkPrevPrev, t_-1=rkPrev, t_i = rkCurrent
		rxPrevPrev = self.pos[0, :]
		ryPrevPrev = self.pos[1, :]

		# TIME STEPPING from intial spatidal and velocity distribution
		vxPrev = self.vel[0, :]
		vyPrev = self.vel[1, :]

		# Verlet scheme for x_i[x_(i-1),x_(i-2)] ]and v_(i-1)[x_i,x_(i-2)], 
		# calculate x_2 before loop, forward Euler
		rxPrev = rxPrevPrev + vxPrev*deltaT
		ryPrev = ryPrevPrev + vyPrev*deltaT
		
		steps = potentialPCB.shape[0]
		stopTime = steps*deltaT*1e6
		
		for s in np.arange(3, steps):
			if s % 100 == 0:
				print 'Step %d, time = %5.2f mus, final time = %5.2f mus' %(s, s*deltaT*1E6, stopTime)
			# adjust potential to current value
			self.acc.fastAdjustAll(potentialPCB[s, :].ctypes.data_as(c_double_p))
			#self.ef.fastAdjustAll(potentialPCB[s, :])
			
			# a(i-1) in Verlet scheme, convert to mm for POTENTIALARRAY object
			xx = 1E3*rxPrev.T
			yy = 1E3*ryPrev.T
			
			# get field gradient at r for all atoms, [POTENTIALARRAY.gradient/.fieldGradient] = V/mm./mm^2 !
			dE = np.zeros((xx.shape[0], 2), dtype=np.double)
			self.acc.getFieldGradient(xx.shape[0], xx.ctypes.data_as(c_double_p), yy.ctypes.data_as(c_double_p), dE.ctypes.data_as(c_double_p))
			#dE = self.ef.getFieldGradient(xx, yy)
			dEx = dE[:, 0]*1E6
			dEy = dE[:, 1]*1E6
			
			#print dEx[0], dEy[0]
			
			# calculate FORCE from current electric field at position of atoms
			fx = -3./2*n*k*a0*e*dEx
			fy = -3./2*n*k*a0*e*dEy
		
			# update POSITION
			rxCurrent = 2.*rxPrev - rxPrevPrev + deltaT**2.*(fx/mass)
			ryCurrent = 2.*ryPrev - ryPrevPrev + deltaT**2.*(fy/mass) 
		
			# update VELOCITY
			vxPrev = (rxCurrent - rxPrevPrev)/(2.*deltaT)
			vyPrev = (ryCurrent - ryPrevPrev)/(2.*deltaT)
		
			# account for particles in electrode and out of potential array
			inElec = self.ef.isElectrode(rxCurrent, ryCurrent)
			inArray = self.ef.inArray(rxCurrent, ryCurrent)
			# since pcb board not specified as eletrode particles below electrodes excluded
			belowPCB = (rxCurrent >= 20.13) & (ryCurrent <= 8.2e-3)
			
			includeIndices = np.where(~inElec & ~belowPCB & inArray)[0]
			
			rxCurrent = rxCurrent[includeIndices]
			ryCurrent = ryCurrent[includeIndices]
			
			rxPrev = rxPrev[includeIndices]
			ryPrev = ryPrev[includeIndices]
			
			rxPrevPrev = rxPrevPrev[includeIndices]
			ryPrevPrev = ryPrevPrev[includeIndices]
			
			vxPrev = vxPrev[includeIndices]
			vyPrev = vyPrev[includeIndices]
			
			# throwing particles out is slower than running with all of them...
			# but of course, they might come back and we won't notice they should have been thrown out
			
			rxPrevPrev = rxPrev[:]
			ryPrevPrev = ryPrev[:]
			
			rxPrev = rxCurrent[:]
			ryPrev = ryCurrent[:]
			
		
		self.vel = np.array([(rxCurrent - rxPrevPrev)/deltaT, (ryCurrent - ryPrevPrev)/deltaT])
		
		self.pos = np.array([rxCurrent, ryCurrent])
		
		#self.acc.free()
		

if __name__ == '__main__':
	from matplotlib import pyplot as plt
	import ctypes
	from ctypes import c_double, c_ulong, c_uint
	c_double_p = ctypes.POINTER(c_double)
	
	from subprocess import call
	target = 'simion_accelerator'
	COMPILE = ['PROF'] # 'PROF', 'FAST', both or neither
	# include branch prediction generation. compile final version with only -fprofile-use
	commonopts = ['-c', '-fPIC', '-Ofast', '-march=native', '-std=c99', '-fno-exceptions', '-fomit-frame-pointer']
	profcommand = ['gcc', '-fprofile-arcs', '-fprofile-generate', target + '.c']
	profcommand[1:1] = commonopts
	fastcommand = ['gcc', '-fprofile-use', target + '.c']
	fastcommand[1:1] = commonopts
	
	print
	print
	print '==================================='
	print 'compilation target: ', target
	if 'PROF' in COMPILE:
		call(profcommand)
		call(['gcc', '-shared', '-fprofile-generate', target + '.o', '-o', target + '.so'])
		print 'COMPILATION: PROFILING RUN'
	if 'FAST' in COMPILE:
		call(fastcommand)
		call(['gcc', '-shared', target + '.o', '-o', target + '.so'])
		print 'COMPILATION: FAST RUN'
	if not ('PROF' in COMPILE or 'FAST' in COMPILE):
		print 'DID NOT RECOMPILE C SOURCE'
	print '==================================='
	print
	print
	
	
	#TODO: parameter handling (!)
	vBeamFwd = 700
	
	n = 30 # stark state
	k = n - 1 # maximum low-field seeking state
	deltaT = 10e-9 # timestep 10ns
	
	xWidth = 1.e-4
	nElectrodes = 12
	field_scale = 10
	
	# POTENTIAL SEQUENCE generated full (in-/outcoupling, guiding/deceleration)
	maxAmp = 20
	vInit = vBeamFwd
	vFinal = vInit # just guiding, no acceleration
	# /mm and /mus
	# deceleration to stop at fixed position
	decelDist = 19.1 # 19.1 for 23 electrodes
	# space beyond minima position after chirp sequence
	# inDist = 2.2mm to first minima with 1/4=Umax, dTotal=21.5mm
	outDist = 21.5 - 2.2 - decelDist # determines stopping position of simulation
	# end point at z position before potential minima diverges
	outDist = 0
	
	# non zero otherwise potential function has NaN at first entry
	incplTime = 0
	outcplTime = 0
	
	# from simulations as maximum value for incoupled atoms
	waveformStartDelay = 18*1E-6;
	
	print '--------------------------------------------------------------------------------'
	print 'STARTING SIMULATION'
	
	flyer = rydberg_flyer()
	flyer.load_fields('./potentials/', field_scale, nElectrodes)
	
	print 'Completed loading %d electrodes' %nElectrodes
	
	flyer.setInitialDistribution2D('LaserExcitedDistribution2DAr.mat', xWidth)
	
	print 'Initial distribution created, ', flyer.pos.shape[1], ' particles'
	print '--------------------------------------------------------------------------------'

	print '\nTrajectory simulation started ...'
	print '--------------------------------------------------------------------------------'
	
	PEE1 = 30.
	PEE2 = 0.
	surfaceBias = 0.

	potentialPCB = generateWaveformPotentials(deltaT,maxAmp,vInit,vFinal,decelDist, incplTime,outcplTime,outDist,waveformStartDelay, PEE1,PEE2,surfaceBias, 23)
	
	# include potentials for photo excitation electrodes, surface and extraction mesh
	# and compensation electrode if applicable
	timeTotal = potentialPCB.shape[1]
	photoElectrode1 = np.zeros((1, timeTotal))
	photoElectrode2 = np.zeros((1, timeTotal))
	# voltage of Rydberg excitation pulsed
	photoElectrode1[:round(waveformStartDelay/deltaT)] = PEE1
	photoElectrode2[:round(waveformStartDelay/deltaT)] = PEE2
	surfaceBias = surfaceBias*np.ones((1, timeTotal))
	extractionMesh = np.zeros((1, timeTotal))
	compensation = np.zeros((1, timeTotal))
	aperture = np.zeros((1, timeTotal))
	potentialPCB = np.ascontiguousarray(np.concatenate((photoElectrode1, photoElectrode2, potentialPCB, surfaceBias, extractionMesh, compensation, aperture)).T.astype(np.double))
	
	# percent of accepted particles of initially excited atoms
	tic = time()
	flyer.fly_atoms(potentialPCB, n, deltaT)
	
	print 'Execution took %5.2f minutes' %((time()-tic)/60.)

	print '--------------------------------------------------------------------------------'
	print '\nTrajectory simulation finished\n'


