import numpy as np
import os

import ctypes
from ctypes import c_double, c_ulong, c_uint
c_double_p = ctypes.POINTER(c_double)


class simion(object):
	def __init__(self, filename, voltages):
		
		directory, fname = os.path.split(filename)
		files = sorted([x for x in os.listdir(directory) if x.startswith(fname) and x.endswith('patxt')])
		
		assert len(files) == len(voltages), 'Incorrect number of potentials specified!'
		
		self.pas = [0]*len(files)
		for i, f in enumerate(files):
			self.pas[i] = patxt(os.path.join(directory, f), voltages[i])
			
		# these are the same for all pas, so they go in here
		self.nx = int(self.pas[0].parameters['nx'])
		self.ny = int(self.pas[0].parameters['ny'])
		self.nz = int(self.pas[0].parameters['nz'])
		
		# make a global electrode map
		self.electrode_map = np.zeros((self.nx, self.ny, self.nz)).astype(np.bool)
		for p in self.pas:
			self.electrode_map |= p.isElectrode
			
		
	def fastAdjust(self, n, v):
		self.pas[n].setVoltage(v)
		
	def fastAdjustAll(self, potentials):
		assert len(potentials) == len (self.pas), 'Number of Voltages passed to fastAdjustAll must equal number of electrodes in potential array'
		for i, p in enumerate(potentials):
			self.pas[i].setVoltage(p)
		
	def getPotential(self, ix, iy, iz):
		value = 0
		for p in self.pas:
			value += p.getPotential(ix, iy, iz)
		return value
		
		
class accelerator(object):
	def __init__(self):
		target = 'simion_accelerator'
		if not os.path.exists(target + '.so') or os.stat(target + '.c').st_mtime > os.stat(target + '.so').st_mtime: # we need to recompile
			from subprocess import call
			
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
				if call(profcommand) != 0:
					print 'COMPILATION FAILED!'
					raise RuntimeError
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
		else:
			print 'library up to date, not recompiling accelerator'
		
		
		self.acc = ctypes.cdll.LoadLibrary('./' + target + '.so')
		
		self.acc.set_npas.argtypes = [c_uint]
		self.acc.set_npas.restype = None
		self.acc.add_pa.argtypes = [c_uint, c_double_p, c_double]
		self.acc.add_pa.restype = None
		self.acc.set_pasize.argtypes = [c_uint, c_uint, c_double, c_double]
		self.acc.set_pasize.restype = None
		self.acc.getFieldGradient.argtypes = [c_uint, c_double_p, c_double_p, c_double_p]
		self.acc.getFieldGradient.restype = None
		self.acc.fastAdjustAll.argtypes = [c_double_p]
		self.acc.fastAdjustAll.restype = None
		
		self.set_npas = self.acc.set_npas
		self.add_pa = self.acc.add_pa
		self.set_pasize = self.acc.set_pasize
		self.getFieldGradient = self.acc.getFieldGradient
		self.fastAdjustAll = self.acc.fastAdjustAll


class EField3D(object):
	def __init__(self, filename, voltages, scale, offset):
		self.simion = simion(filename, voltages)
		
		self.x0 = offset[0]
		self.y0 = offset[1]
		self.z0 = offset[2]
		
		self.dx = 1./scale
		self.dy = 1./scale
		self.dz = 1./scale
		
		self.nx = self.simion.nx
		self.ny = self.simion.ny
		self.nz = self.simion.nz
		
		self.xmax = self.nx*self.dx + self.x0;
		self.ymax = self.ny*self.dy + self.y0;
		self.zmax = self.nz*self.dz + self.z0;
		self.xmin = self.x0;
		self.ymin = self.y0;
		self.zmin = -self.zmax;

	def getPotential(self, x, y, z):
		# POTENTIAL Get the magnitude of the electric potential.
		#	 Calculate the magnitude of the electrostatic potential at
		#	 coordinates x, y, z by interpolating the scaled fields from
		#	 each electrode. If r or x is outside the boundary, the
		#	 value at the boundary is returned.
		
		# Fractional potential array index.
		ixf = (x-self.x0)/self.dx - 1
		iyf = (y-self.y0)/self.dy - 1
		izf = abs(z-self.z0)/self.dz - 1
		
		# Integer part of potential array index.
		ix = np.where(np.ceil(ixf) < self.nx - 1, np.ceil(ixf), self.nx-2).astype(np.int)
		iy = np.where(np.ceil(iyf) < self.ny - 1, np.ceil(iyf), self.ny-2).astype(np.int)
		iz = np.where(np.ceil(izf) < self.nz - 1, np.ceil(izf), self.nz-2).astype(np.int)
		
		# Calculate distance of point from gridlines.
		#		 xd = (ixf - floor(ixf)).*this.dx;
		#		 yd = (iyf - floor(iyf)).*this.dy;
		#		 zd = (izf - floor(izf)).*this.dz;
		xd = (ixf - np.floor(ixf))
		yd = (iyf - np.floor(iyf))
		zd = (izf - np.floor(izf))
		
		Q111 = self.simion.getPotential(ix	, iy	, iz	)
		Q112 = self.simion.getPotential(ix	, iy	, iz+1)
		Q121 = self.simion.getPotential(ix	, iy+1, iz	)
		Q122 = self.simion.getPotential(ix	, iy+1, iz+1)
		Q211 = self.simion.getPotential(ix+1, iy	, iz	)
		Q212 = self.simion.getPotential(ix+1, iy	, iz+1)
		Q221 = self.simion.getPotential(ix+1, iy+1, iz	)
		Q222 = self.simion.getPotential(ix+1, iy+1, iz+1)
		
		i1 = (xd*Q211 + (1-xd)*Q111)
		i2 = (xd*Q221 + (1-xd)*Q121)
		j1 = (xd*Q212 + (1-xd)*Q112)
		j2 = (xd*Q222 + (1-xd)*Q122)
		
		k1 = (yd*i2 + (1-yd)*i1)
		k2 = (yd*j2 + (1-yd)*j1)
		
		return (zd*k2 + (1-zd)*k1)
		
	def getField3(self, pos):
		# GRADIENT Calculate the potential gradient at r,x.
		# The gradient is calculated from the central-difference
		# approximation finite differences.
		x = pos[:, 0]
		y = pos[:, 1]
		z = pos[:, 2]
		
		hx = self.dx/2.0
		hy = self.dy/2.0
		hz = self.dz/2.0
		
		px2 = self.getPotential(x+hx, y, z)
		px1 = self.getPotential(x-hx, y, z)
		py2 = self.getPotential(x, y+hy, z)
		py1 = self.getPotential(x, y-hy, z)
		pz2 = self.getPotential(x, y, z+hz)
		pz1 = self.getPotential(x, y, z-hz)
		
		dfx = (px2-px1)/self.dx
		dfy = (py2-py1)/self.dy
		dfz = (pz2-pz1)/self.dz
		return np.array([dfx, dfy, dfz]).T
		

	def inArray3(self, pos):
		x = pos[:,0]
		y = pos[:,1]
		z = pos[:,2]
		return (x > self.xmin) & (x < self.xmax) & (y > self.ymin) & (y < self.ymax) & (z > self.zmin) & (z < self.zmax)
		
	def fastAdjust(self, n, v):
		self.simion.fastAdjust(n, v)
		
	def isElectrode3(self, pos):
		# ISELECTRODE Test if point r, x is within an electrode.
		# Returns true if (r, x) is inside an electrode.
		
		x = pos[:, 0]
		y = pos[:, 1]
		z = pos[:, 2]
		assert y.shape == x.shape, 'r and x arrays are different sizes.'
		
		# Integer part of potential array index.
		ixf = (x-self.x0)/self.dx - 1
		iyf = (y-self.y0)/self.dy - 1
		izf = abs(z-self.z0)/self.dz - 1
		
		# Integer part of potential array index.
		ix = np.where(np.ceil(ixf) < self.nx - 1, np.ceil(ixf), self.nx-2).astype(np.int)
		iy = np.where(np.ceil(iyf) < self.ny - 1, np.ceil(iyf), self.ny-2).astype(np.int)
		iz = np.where(np.ceil(izf) < self.nz - 1, np.ceil(izf), self.nz-2).astype(np.int)
		
		
		ix[ix < 0] = 0
		iy[iy < 0] = 0
		iz[iz < 0] = 0
		
		return self.simion.electrode_map[ix, iy, iz].flatten()

class EField2D(simion):
	def __init__(self, filename, voltages, scale, use_accelerator = False):
		super(EField2D, self).__init__(filename, voltages)
		
		self.dx = 1./scale
		self.dr = 1./scale
		
		self.nr = self.ny # treat y direction as r
		
		self.xmax = self.nx*self.dx
		self.rmax = self.nr*self.dr
		
		if use_accelerator:
			a = accelerator()
			a.set_npas(len(voltages))
			a.set_pasize(self.nx, self.nr, self.dx, self.dr)
			for n, p in enumerate(self.pas):
				a.add_pa(n, p.potential.ctypes.data_as(c_double_p), 0)
			
			self.fastAdjustAll = lambda V: a.fastAdjustAll(V.ctypes.data_as(c_double_p))
			def helper(xx, yy):
				dE = np.zeros((xx.shape[0], 2), dtype=np.double)
				a.getFieldGradient(xx.shape[0], xx.ctypes.data_as(c_double_p), yy.ctypes.data_as(c_double_p), dE.ctypes.data_as(c_double_p))
				return dE
			self.getFieldGradient = helper
			self.getField = None 			# to prevent anyone from accidentally trying to call these
			self.getField3 = None
			self.getPotential = None

		
	def getPotential(self, x, r):
		r = abs(r)
		
		ixf = x/self.dx - 1
		irf = r/self.dr - 1
		
		# Integer part of potential array index.
		ir = np.where(np.ceil(irf) < self.nr - 1, np.ceil(irf), self.nr-2).astype(np.int)
		ix = np.where(np.ceil(ixf) < self.nx - 1, np.ceil(ixf), self.nx-2).astype(np.int)
		
		ir[ir < 0] = 0
		ix[ix < 0] = 0
		
		# if isscalar(r) && isscalar(x)
		Q11 = super(EField2D, self).getPotential(ix,	 ir,	 0)
		Q12 = super(EField2D, self).getPotential(ix+1, ir,	 0)
		Q21 = super(EField2D, self).getPotential(ix,	 ir+1, 0)
		Q22 = super(EField2D, self).getPotential(ix+1, ir+1, 0)
		
		# Calculate distance of point from gridlines.
		r1 = (irf - np.floor(irf))*self.dr
		r2 = self.dr-r1
		x1 = (ixf - np.floor(ixf))*self.dx
		x2 = self.dx-x1
		
		# Linear interpolation function.
		return ((Q11*r2*x2) + (Q21*r1*x2) + (Q12*r2*x1) + (Q22*x1*r1))/(self.dx*self.dr)
		
	
	def getField3(self, pos):
		# GRADIENT Calculate the potential gradient at r,x.
		#	 The gradient is calculated from the centred-difference
		#	 approximation finite differences.
		
		r = np.sqrt(pos[:, 1]**2+pos[:, 2]**2)
		x = pos[:, 0]
		
		hr = self.dr/2.
		hx = self.dx/2.
		
		p1 = self.getPotential(x-hx, r)
		p2 = self.getPotential(x+hx, r)
		p3 = self.getPotential(x, r-hr)
		p4 = self.getPotential(x, r+hr);
		
		dfr = (p4-p3)/self.dr
		dfx = (p2-p1)/self.dx
		
		dfy = dfr*np.sin(np.arctan2(pos[:, 1], pos[:, 2]))
		dfz = dfr*np.cos(np.arctan2(pos[:, 1], pos[:, 2]))
		return np.array([dfx, dfy, dfz]).T
		
	
	def getField(self, x, r):
		# GRADIENT Calculate the potential gradient at r,x.
		#	 The gradient is calculated from the centred-difference
		#	 approximation finite differences.
		
		hx = self.dx/2.
		hr = self.dr/2.
		
		
		p1 = self.getPotential(x-hx, r)
		p2 = self.getPotential(x+hx, r)
		print 'p2: %25.23E' %p2[0]
		p3 = self.getPotential(x, r-hr)
		p4 = self.getPotential(x, r+hr);
		
		dfx = (p2-p1)/self.dx
		dfr = (p4-p3)/self.dr
		
		return np.array([dfx, dfr]).T
	
	def getFieldGradient(self, x, r):
		# based on following formula:
		# Fx: x-component of force
		# Ex: x-component of field
		# U: potential
		# Fx \propto dx |E| = dx sqrt(Ex^2+Ey^2+Ez^2)
		#	   = (Ex dx Ex + Ey dx Ey + Ez dx Ez)/|E|
		# now dx \vec(E) = (E(x+hx, y, z) - E(x-hx, y, z))/this.dx
		# in the code E(x+hx...) is called dx2, minus version dx1
		# other compnents equivalently
		hx = self.dx/2.
		hr = self.dr/2.
		
		E0 = self.getField(x, r)
		normE = np.sqrt(np.sum(E0**2, 1))
		
		# otherwise return is NaN
#		if normE == 0:
#			raise RuntimeError
		
		dx2 = self.getField(x+hx, r)
		dx1 = self.getField(x-hx, r)
		dEx = np.diag(E0.dot((dx2.T-dx1.T)/self.dx))/normE
		
		dy2 = self.getField(x, r+hr)
		dy1 = self.getField(x, r-hr)
		dEy = np.diag(E0.dot((dy2.T-dy1.T)/self.dr))/normE
		return np.array([dEx, dEy]).T


	def inArray3(self, pos):
		r = np.sqrt(pos[:,1]**2+pos[:,2]**2)
		x = pos[:,0]
		return self.inArray(x, r)
	
	def inArray(self, x, r):
		return (r >= 0) & (x > 0) & (r < self.rmax) & (x < self.xmax)
		
	def isElectrode3(self, pos):
		# ISELECTRODE Test if point r, x is within an electrode.
		#	 Returns true if (r, x) is inside an electrode.
		
		r = np.sqrt(pos[:, 1]**2 + pos[:, 2]**2)
		x = pos[:, 0]
		
		return self.isElectrode(x, r)
	
	def isElectrode(self, x, r):
		assert r.shape == x.shape, 'r and x arrays are different sizes'
		
		# Fractional potential array index.
		irf = r/self.dr
		ixf = x/self.dx
		
		# Integer part of potential array index.
		ir = np.where(np.ceil(irf) < self.nr - 1, np.ceil(irf), self.nr-2).astype(np.int)
		ix = np.where(np.ceil(ixf) < self.nx - 1, np.ceil(ixf), self.nx-2).astype(np.int)
		
		ir[ir < 0] = 0
		ix[ix < 0] = 0
		
		return self.electrode_map[ix, ir].flatten()

		
class patxt(object):
	def __init__(self, filename, V):
		self.voltage = V
		self.filename = filename
		
		
		self.parameters = {}
		# read header data
		with open(filename, 'rb') as ifile:
			# read header, save parameters in dict
			for l in range(18):
				line = ifile.readline()
				if line.startswith(' '):
					self.parameters[line.split()[0]] = line.split()[1]
			
		# make some parameters accessible by shorthand
		nx = int(self.parameters['nx'])
		ny = int(self.parameters['ny'])
		nz = int(self.parameters['nz'])
		
		# now read the actual data
		if os.path.isfile(filename + '.npy'): # TODO: check date, not just existence
			print 'loading ', filename, 'from cache'
			data = np.load(filename + '.npy')
		else:
			data = np.genfromtxt(filename, delimiter = ' ', skip_header = 18, skip_footer = 2, usecols = (3, 4))
			np.save(filename, data)
		
		assert data.shape[0] == nx*ny*nz
		
		self.isElectrode = data[:, 0].reshape((nx, ny, nz), order='F').astype(np.bool)
		self.potential = data[:, 1].reshape((nx, ny, nz), order='F').astype(np.double)
		self.potential = np.ascontiguousarray(self.potential)
		self.potential /= 10000.0
		
	def setVoltage(self, V):
		self.voltage = V
	
	def getPotential(self, ix, iy, iz):
		return self.voltage*self.potential[ix, iy, iz]

if __name__ == '__main__':
	ef1 = EField3D('Simion Fields/quadtrap2/quadtrap2', [1, 1, 1, 1, 0], 1/5e-4, np.array([-67, -12.75, 30.5])*1e-3);
	ef2 = EField2D('Simion Fields/tof_test4/tof_test3', [-1900, 100, 0], 1/1e-3);

