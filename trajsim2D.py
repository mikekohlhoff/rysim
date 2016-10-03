import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.io import loadmat
from time import time, strftime
from matplotlib import pyplot as plt
from WaveformPotentials import WaveformPotentials23Elec

import sys
import os

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from simioniser.EField2D import EField2D
from simioniser.EField3D import EField3D

# CONSTANTS
# mass of hydrogen
mass = 1.6735327160314e-27
# Boltzmann constant / J/K
kB = 1.380648813e-23
# Bohr radius / m
a0 = 5.291772109217e-11
# elementary charge / C
e = 1.60217656535e-19

class RSDTrajSim(object):
        def __init__(self):
                pass
        
        def setInitialDistribution2D(self, filename, xWidth):
                A = loadmat(filename)
                
                # this is still a 3D distribution, so for the 2D simulation we're interested in here
                # select a central slice only, and disregard the x-axis afterwards
                mask = (A['rLasersIntercept'][0] > -xWidth/2) & (A['rLasersIntercept'][0] < xWidth/2)
                self.pos = np.flipud(A['rLasersIntercept'][1:3, mask])
                self.vel = np.flipud(A['vLasersIntercept'][1:3, mask])
                #self.vel[0,:] = self.vel[0,:]*np.cos(0.05) - self.vel[1,:]*np.sin(0.05)
                #self.vel[1,:] = self.vel[1,:]*np.cos(0.05) + self.vel[0,:]*np.sin(0.05)

                
                # shift everything by rydberg excitation point
                rydbergExcitationPoint = np.array([[102, 87.5]])*np.array([[self.ef.dx, self.ef.dr]])*1E-3
                self.pos += rydbergExcitationPoint.T

                self.initnum = self.pos.shape[1]

        def loadFields(self, folder, scale, nElectrodes):
                self.ef = EField2D(folder, [0]*nElectrodes, scale, use_accelerator=True)
                
        
        def collision(self, pos, ef):
        # atoms hit gridpoint declared as electrode
                collision_index = np.ones_like(pos)
                for f in ef:
                        index = f.isElectrode(pos)
                        collision_index[index, :] = 0
                
                return collision_index
        
        def potSequence(self,mode,looparg,deltaT,vInit,vFinal,incplTime,outcplTime,maxAmp,elecSelect,wfstartdelay,PEE1,PEE2,surf,mesh,comp,aperture):
            # build list of potential and time sequences to use for propagation, return list of arrays
            # Electrode layout in Simion:
            # 1: PEE1 low
            # 2: PEE2 up
            # 3-8: waveform potentials
            # 9: strip compensation
            # 10: aperture
            # 11: surface
            # 12: mesh
            
          # deceleration to stop at fixed position
            decelDist = 19.1 # 19.1 for 23 electrodes
            # space beyond minima position after chirp sequence
            # inDist = 2.2mm to first minima with 1/4=Umax, dTotal=21.5mm
            outDist = 21.5 - 2.2 - decelDist # determines stopping position of simulation

            # let simulations run further than end of wf sequence
            tend = 500E-6

            wfarr = []
            for p in looparg: 
                pcbpot = WaveformPotentials23Elec()

                # pcb sequence start delay
                if mode == 1:
                    wfstartdelay = p
                elif mode == 2:
                    incplTime = p
                elif mode == 3:
                    outcplTime = p
                elif mode == 4:
                    maxAmp = p/2.
                elif mode == 5: pass

                pcbpot.generate(deltaT, vInit, vFinal, incplTime, outcplTime, maxAmp, decelDist, elecSelect)
                wfarr.append(pcbpot.buildArray(deltaT, wfstartdelay, PEE1, PEE2, surf, mesh, comp, aperture,tend))

                #pcbpot.plot()

            return wfarr

        def propagateAtoms(self, potentialPCB, n, deltaT):
        # propagate atoms with position Verlet
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
               
                # record excluded particles [[posx,posy,vx,vy,t]]
                recPart = np.empty([0,5])

                for s in np.arange(3, steps):
                        
                        if s % int(round(1E-6/deltaT)) == 0:
                                print 'Step %d, time = %5.2f mus' %(s, s*deltaT*1E6)
                        # adjust potential to current value
                        self.ef.fastAdjustAll(potentialPCB[s, :])
                        
                        # a(i-1) in Verlet scheme, convert to mm for POTENTIALARRAY object
                        xx = 1E3*rxPrev.T
                        yy = 1E3*ryPrev.T
                        
                        # get field gradient at r for all atoms, [POTENTIALARRAY.gradient/.fieldGradient] = V/mm./mm^2 !
                        dE = self.ef.getFieldGradient(xx, yy)*1E6
                        dEx = dE[:, 0]
                        dEy = dE[:, 1]
                        
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
                        inElec = self.ef.isElectrode(rxCurrent*1E3, ryCurrent*1E3)

                        inArray = self.ef.inArray(rxCurrent*1E3, ryCurrent*1E3)

                        # since pcb board not specified as eletrode particles below electrodes excluded
                        belowPCB = (rxCurrent >= 20.1) & (ryCurrent <= 8.2e-3)
                        
                        # included particles that further propagate
                        includeIndices = np.where(~inElec & ~belowPCB & inArray)[0]
                        # excluded particles recorded with time/pos/vel
                        excluded = np.where(inElec | belowPCB | ~inArray)[0]
                        
                        if len(excluded) > 0:
                            exPart = np.hstack((np.reshape(rxCurrent[excluded],(-1,1)),np.reshape(ryCurrent[excluded],(-1,1)), \
                                                           np.reshape(vxPrev[excluded],(-1,1)), np.reshape(vyPrev[excluded],(-1,1)), \
                                                           np.full((len(excluded),1),s*deltaT,dtype=float)))
                            recPart = np.append(recPart, exPart, axis=0)

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
                        
                        # abort propagation when all particles are excluded
                        if len(rxCurrent) == 0: break
                
                return recPart

if __name__ == '__main__':
        from matplotlib import pyplot as plt
        
        # chip width 8mm, initial cloud has about ~100k particles, choice of width maskes initial amount for propagation
        xWidth = 6e-3
        nElectrodes = 12
        field_scale = 10

        # loop over parameters
        # mode 1: waveform delay
        # mode 2: incoupling time
        # mode 3: outcoupling time
        # mode 4: potential maximum magnitude
        # mode 5: pqn
        mode = 1

        # seeding gas 
        gas = 'He'

       
        print '--------------------------------------------------------------------------------'
        print 'STARTING SIMULATION'
        
        propagator = RSDTrajSim()
        propagator.loadFields('./potentials/full/', field_scale, nElectrodes)
        
        print 'Completed loading %d electrodes' %nElectrodes
        
        propagator.setInitialDistribution2D('LaserExcitedDistribution2D' + gas + '.mat', xWidth)
        
        print 'Initial distribution created, ', propagator.initnum, ' particles'
        print '--------------------------------------------------------------------------------'
        
        print '\nTrajectory simulation started ...'
        print '--------------------------------------------------------------------------------'

        # He: 1600, Ne: 975, Ar: 700
        if gas == 'He': vBeamFwd = 1600
        elif gas == 'Ne': vBeamFwd = 975
        elif gas == 'Ar':vBeamFwd = 700
 
        # Stark State
        n = 35

        # timestep in propagation 1ns, exp 50ns @ 20MHz TODO
        deltaT = 10./1E9
        
        PEE1 = 30.
        PEE2 = 0.
        surf = 0.
        mesh = 0.
        comp = 0.
        aperture = 0.
 
        # select which electrode pair has first max pot - position of minima closest to start of chip, (1,4),(2,5),(3,6)
        elecSelect = '(1,4)'

        # POTENTIAL SEQUENCE generated full (in-/outcoupling, guiding/deceleration), amplitude = 0 - max, -max/2 - max/2 in cos(phi)
        maxAmp = 80
        vInit = vBeamFwd
        vFinal = vInit # just guiding, no acceleration
        # /mm and /mus
        
        # non zero otherwise potential function has NaN at first entry
        incplTime = 1
        outcplTime = 2.5
        
        # from simulations as maximum value for guided atoms arriving at detector
        if gas == 'He': wfstartdelay = 8.3*1E-6
        elif gas == 'Ne': wfstartdelay = 14.5*1E-6
        if gas == 'Ar': wfstartdelay = 17.3*1E-6


        # if not chosen as parameter to loop over value above used as defaults
        if mode == 0:
            looparg = [0]
            loopparam = 'None'
        elif mode == 1:
            looparg = np.arange(0,30.5,0.5)*1E-6
            loopparam = 'wfstartdelay'
        elif mode == 2:
            looparg = np.arange(0,10.1,.1)*1E-6
            loopparam = 'incpltime'
        elif mode == 3:
            looparg = np.arange(0,10.1,.1)*1E-6
            loopparam = 'outcpltime'
        elif mode == 4:
            looparg = np.arange(0,201,1)
            loopparam = 'potential magnitude'
        elif mode == 5:
            looparg = np.arange(20,61,1)
            loopparam = 'pqn'

        # input: potSequence(self, mode,looparg,deltaT,vInit,vFinal,incplTime,outcplTime,maxAmp,elecSelect,p,PEE1,PEE2,surf,mesh,comp,aperture):
        wfarr = propagator.potSequence(mode,looparg,deltaT,vInit,vFinal,incplTime,outcplTime,maxAmp/2.,elecSelect,wfstartdelay,PEE1,PEE2,surf,mesh,comp,aperture)

        tic = time()
        for j in range(len(looparg)):
            if mode == 5: pqn = looparg[j]
            else: pqn = n
            recparts = propagator.propagateAtoms(wfarr[j], n, deltaT)
            np.savetxt('./dataout/HeGuidingAmp80/prop_particles_' + gas + '_' + strftime('%y%m%d%H%M') + '_' + 'n' + str(pqn) + '_' + loopparam + '_' + str(looparg[j]), recparts, delimiter='\t', newline='\n')
            print 'Iteration for argument: ' + str(looparg[j])
            print '--------------------------------------------------------------------------------'

        print 'Execution took %5.2f minutes' %((time()-tic)/60.)
        print '--------------------------------------------------------------------------------'
        print '\nTrajectory simulation finished\n'
        print '--------------------------------------------------------------------------------'
