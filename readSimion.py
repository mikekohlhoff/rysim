import numpy as np
import os

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
		self.electrode_map = np.zeros((self.nx, self.ny, self.nz)).astype(np.int)
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
		
	def getField(self, pos):
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
		

	def inArray(self, pos):
		x = pos[:,0]
		y = pos[:,1]
		z = pos[:,2]
		return (x > self.xmin) & (x < self.xmax) & (y > self.ymin) & (y < self.ymax) & (z > self.zmin) & (z < self.zmax)
		
	def fastAdjust(self, n, v):
		self.simion.fastAdjust(n, v)
		
	def isElectrode (self, pos):
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
		
		
		# return self.iselec(sub2ind(size(this.elec), ix, iy, iz)) == 1;
		return np.where(self.simion.electrode_map[ix, iy, iz])



class EField2D(simion):
	def __init__(self, filename, voltages, scale):
		#self.simion = simion(filename, voltages)
		super(EField2D, self).__init__(filename, voltages)
		
		self.dx = 1./scale
		self.dr = 1./scale
		
		self.nr = self.ny # treat y direction as r
		
		self.xmax = self.nx*self.dx
		self.rmax = self.nr*self.dr
		
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
		
	
	def getField(self, pos):
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
		
	def getField2(self, x, r):
		# GRADIENT Calculate the potential gradient at r,x.
		#	The gradient is calculated from the centred-difference
		#	approximation finite differences.
		# here we do an integrated version not relying on getPotential, to reduce double evaluation of points
		
		hr = self.dr/2.
		hx = self.dx/2.
		
		r = abs(r)
		
		# for r, x+hx:
		xp = x + hx
		xm = x - hx
		rp = r + hr
		rm = r - hr
		
		
		ixp = xp/self.dx - 1
		ixm = xm/self.dx - 1
		ix0 = x/self.dx - 1
		irp = rp/self.dr - 1
		irm = rm/self.dr - 1
		ir0 = r/self.dr - 1
		
		# Integer part of potential array index.
		ix0 = np.where(np.ceil(ix0) < self.nx - 1, np.ceil(ix0), self.nx-2).astype(np.int)
		ixp = np.where(np.ceil(ixp) < self.nx - 1, np.ceil(ixp), self.nx-2).astype(np.int)
		ixm = np.where(np.ceil(ixm) < self.nx - 1, np.ceil(ixm), self.nx-2).astype(np.int)
		ir0 = np.where(np.ceil(ir0) < self.nr - 1, np.ceil(ir0), self.nr-2).astype(np.int)
		irp = np.where(np.ceil(irp) < self.nr - 1, np.ceil(irp), self.nr-2).astype(np.int)
		irm = np.where(np.ceil(irm) < self.nr - 1, np.ceil(irm), self.nr-2).astype(np.int)
		
		ix0[ix0 < 0] = 0
		ixp[ixp < 0] = 0
		ixm[ixm < 0] = 0
		ir0[ir0 < 0] = 0
		irp[irp < 0] = 0
		irm[irm < 0] = 0
		
		
		# smallest possible value is ixm, biggest value ixp + 1 (ir analogous)
		# but doing this right is quite tricky when acting on arrays
		# in single-particle this would be fairly trivial
		
		Q11p0 = super(EField2D, self).getPotential(ixp,	 ir0,	 0)
		Q12p0 = super(EField2D, self).getPotential(ixp+1, ir0,	 0)
		Q21p0 = super(EField2D, self).getPotential(ixp,	 ir0+1, 0)
		Q22p0 = super(EField2D, self).getPotential(ixp+1, ir0+1, 0)
		
		Q11m0 = super(EField2D, self).getPotential(ixm,	 ir0,	 0)
		Q12m0 = super(EField2D, self).getPotential(ixm+1, ir0,	 0)
		Q21m0 = super(EField2D, self).getPotential(ixm,	 ir0+1, 0)
		Q22m0 = super(EField2D, self).getPotential(ixm+1, ir0+1, 0)
		
		Q110p = super(EField2D, self).getPotential(ix0,	 irp,	 0)
		Q120p = super(EField2D, self).getPotential(ix0+1, irp,	 0)
		Q210p = super(EField2D, self).getPotential(ix0,	 irp+1, 0)
		Q220p = super(EField2D, self).getPotential(ix0+1, irp+1, 0)
		
		Q110m = super(EField2D, self).getPotential(ix0,	 irm,	 0)
		Q120m = super(EField2D, self).getPotential(ix0+1, irm,	 0)
		Q210m = super(EField2D, self).getPotential(ix0,	 irm+1, 0)
		Q220m = super(EField2D, self).getPotential(ix0+1, irm+1, 0)
		
		# Calculate distance of point from gridlines.
		x10 = (ix0 - np.floor(ix0))
		x20 = 1-x10
		x1p = (ixp - np.floor(ixp))
		x2p = 1-x1p
		x1m = (ixm - np.floor(ixm))
		x2m = 1-x1m
		
		r10 = (ir0 - np.floor(ir0))
		r20 = 1-r10
		r1p = (irp - np.floor(irp))
		r2p = 1-r1p
		r1m = (irm - np.floor(irm))
		r2m = 1-r1m
		
		
		p1 = ((Q11m0*r20*x2m) + (Q21m0*r10*x2m) + (Q12m0*r20*x1m) + (Q22m0*x1m*r10))
		p2 = ((Q11p0*r20*x2p) + (Q21p0*r10*x2p) + (Q12p0*r20*x1p) + (Q22p0*x1p*r10))
		p3 = ((Q110m*r2m*x20) + (Q210m*r1m*x20) + (Q120m*r2m*x10) + (Q220m*x10*r1m))
		p4 = ((Q110p*r2p*x20) + (Q210p*r1p*x20) + (Q120p*r2p*x10) + (Q220p*x10*r1p))
		
		dfr = (p4-p3)/self.dr
		dfx = (p2-p1)/self.dx
		
		return np.array([dfx, dfr]).T

		
	
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
		
		tmp = dx2 - dx1
		
		print 'individuals: %25.23E, %25.23E, %25.23E, %25.23E' %(tmp[0, 0], E0[0, 0], tmp[0, 1], E0[0, 1])
		
		print 'manual: %25.23E' %(tmp[0, 0]*E0[0, 0] + tmp[0, 1]*E0[0, 1])
		print 'dot: %25.23E' %(tmp.dot(E0.T)[0, 0])
		
		dEx = np.diag(E0.dot((dx2.T-dx1.T)/self.dx))/normE
		print 'dEx: %15.10E' %(dEx[0])
		
		dy2 = self.getField(x, r+hr)
		dy1 = self.getField(x, r-hr)
		dEy = np.diag(E0.dot((dy2.T-dy1.T)/self.dr))/normE
		return np.array([dEx, dEy]).T


	def inArray(self, pos):
		r = np.sqrt(pos[:,1]**2+pos[:,2]**2)
		x = pos[:,0]
		return self.inArray(x, r)
	
	def inArray(self, x, r):
		return (r >= 0) & (x > 0) & (r < self.rmax) & (x < self.xmax)
		
	def isElectrode(self, pos):
		# ISELECTRODE Test if point r, x is within an electrode.
		#	 Returns true if (r, x) is inside an electrode.
		
		r = np.sqrt(pos[:, 1]**2 + pos[:, 2]**2)
		x = pos[:, 1]
		
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
		
		self.isElectrode = data[:, 0].reshape((nx, ny, nz), order='F').astype(np.int)
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

