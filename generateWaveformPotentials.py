import numpy as np

def generateWaveformPotentials(timeStep,maxAmp,vInit,vFinal,decelDist,incplTime, outcplTime,outDist,waveformStartDelay, PEE1,PEE2,surfaceBias, nElectrodes):
	# Build ELECTRODE POTENTIALS piece-wise for incoupling, chirp, outcoupling.
	# Oscillation frequency omega building phase, constant (inc/outc) or non-constant 
	# over time (chirp); t0=0, t1=incplTime, t2=incplTime+decelTime, t3=relative time(end)-outcplTime, 
	# t4=time(end)
	# position of first fixed in z-direction after set incoupling time
	# chirp chosen to end at fixed position before field minima diverges

	# time inputs in 
	# distance between field minima, 3*center-to-center spacing of electrodes
	dmin = 3E-3
	phaseOffset = 2./3*np.pi*np.array([[0, 2, 1, 0, 2, 1]]).T
	
	# first minima closer to excitation electrodes
	# phaseOffset = 2/3*pi*[2;1;0;2;1;0];
	# time = time*1E-6;
	incplTime *= 1E-6
	outcplTime *= 1E-6
	# decrease potential after set position for minima
	outcplTime = outDist*1E-3/vFinal
	decelDist *= 1E-3
	# 21 electrodes assumed with 0.5mm width and 0.5mm apart w/o margins
	chipLength = (nElectrodes + 0.5)*1E-3;
	
	# check available deceleration path length over chip and cancel if neccessary 
	distanceTotal = vInit*incplTime + vFinal*outcplTime + decelDist
	# time passed during a-/decceleration for given length decelDist
	decelTime = 2*decelDist/(vInit+vFinal)
	# make calculated time comparable to time step in time vector
	p = np.ceil(np.log10(1./timeStep))
	# decelTime = round(decelTime*10^p)/(10^p);
	incplTime = np.floor(incplTime*1E9)/1E9
	outcplTime = np.floor(outcplTime*1E9)/1E9
	decelTime = np.ceil(decelTime*1E9)/1E9;
	timeTotal = decelTime + outcplTime
	time = np.arange(0, timeTotal, timeStep)
	
	assert distanceTotal <= chipLength, 'Chip length possibly insufficient for combination of in- and outcoupling times and/or intended deceleration path length in z-direction, required flightpath = %.2f mm, available chip length = %.2f mm' %(1e3*distanceTotal, 1e3*chipLength)
	
	if True: # TODO: verbose == True
		print 'Time of flight for completion of given acceleration distance: %.2f mus' % (1e6*decelTime)
		print 'Required time of flight complete = %.2f mus' % (timeTotal*1E6)
		print 'Path length for incoupling time of flight %.2f mm (1.7mm to 1st min)' % (vInit*incplTime*1E3)
		print 'Path length for outcoupling time of flight = %.2f mm' %(vInit*outcplTime*1E3)
		print 'Required path length complete = %.2f mm (%.2f mm total chip length available)' %(distanceTotal*1E3, chipLength*1E3)
	
	# 1. INCOUPLING, linear increase in amplitude, constant omega
	# compare in ns to avoid rounding errors
	timeInRev = time[time*1E9 <= incplTime*1E9]
	timeIn = -time[time*1E9 <= incplTime*1E9][::-1]
	omegaConst = 2*np.pi*vInit/dmin
	phaseIn = omegaConst*timeIn
	ampIn = maxAmp*timeInRev/incplTime
	ampIn[ampIn > maxAmp] = 0
	# plot(ampIn.T)
	phaseIn = phaseIn + phaseOffset
	newPotIn = ampIn*np.array([[-1, 1, -1, 1, -1, 1]]).T*(np.cos(phaseIn)+1)

	# 2. electrode potentials with LINEAR CHIRP
	# decelTime + incpltime = t2, t=(ti-t1)
	timeChirp = time[time*1E9 < (decelTime-1*timeStep)*1E9] + timeStep
	freqSlope = (np.pi/(2*decelDist*dmin)*(vFinal**2-vInit**2))
	phaseChirp = (freqSlope*(timeChirp) + 2*np.pi*vInit/dmin)*(timeChirp)
	chirpOffset = 2*np.pi/dmin*vInit*(incplTime + 0*timeStep)
	phase = phaseChirp + chirpOffset
	# keep minimum at position for 1/4=Umax
	phase = phaseChirp
	phase = np.tile(phase, (6, 1)) + phaseOffset
	newPotChirp = maxAmp*np.array([[-1, 1, -1, 1, -1, 1]]).T*(np.cos(phase)+1)
	
	# 3. CONSTANT frequency after chirp completed 
	timePostChirp = time[time*1E9 < (time[-1]*1E9 - incplTime*1E9 - decelTime*1E9 - 0*timeStep*1E9)]
	# don't account for incoupling time
	timePostChirp = time[time*1E9 < (time[-1]*1E9 - decelTime*1E9 - 0*timeStep*1E9)]
	postOffset = 2*np.pi/dmin*(vInit*(incplTime*0 + decelTime + 0*timeStep) + (vFinal**2-vInit**2)/(4*decelDist)*(decelTime)**2)
	phasePostChirp = 2*np.pi/dmin*vFinal*timePostChirp + postOffset;
	phasePostChirp = phasePostChirp + phaseOffset
	newPotPostChirp = maxAmp*np.array([[-1, 1, -1, 1, -1, 1]]).T*(np.cos(phasePostChirp)+1);
		
	# 4. OUTCOUPLING, linear decrease in amplitude for omega1
	timeOut = timePostChirp + decelTime + incplTime + timeStep
	timeOut = timePostChirp + decelTime + timeStep
	ampOut = (time[-1] - timeOut)/outcplTime
	ampOut[ampOut > 1] = 1
	newPotPostChirp = newPotPostChirp*ampOut

	# concetenate different electrode potential phases
	newPot = np.concatenate((newPotIn, newPotChirp, newPotPostChirp), 1)
	
#	if(plotMode == 1)
#			timeTotalPlot = incplTime + decelTime + outcplTime
#			timePlot = 1E6*timeStep*[0:1:length(newPot)-1]
#			# plotting and saving waveform to file 
#			clf(1)
#			setFigure(figureNumber)
#			set(gcf,'DefaultAxesColorOrder',[0 0 1;1 0 0;0 0 0], ...
#		  	'DefaultAxesLineStyleOrder','-|--')
#			plot(timePlot,newPot, 'LineWidth', 1, 'MarkerSize', 1)
#			hold on
#	#  	set(gca,'XTick',0:1:time(end)*1E6+1)
#			xlabel('Time (\mus)', 'FontSize', 12); ylabel('Electrode potential (V)', 'FontSize', 12)
#			hleg = legend('1', '2', '3', '4', '5', '6');
#			set(hleg, 'FontSize', 12, 'Box', 'off', 'Location', 'NorthEast')
#		  title('PCB potentials', 'FontSize', 12)
#	 	# set time markers
#	 	time1 = incplTime*1E6;
#			line([time1 time1],[-40 40], 'LineStyle', '--', 'LineWidth', 1, 'Color', 'k');
#			time2 = time1 + decelTime*1E6;
#			line([time2 time2],[-40 40], 'LineStyle', '--', 'LineWidth', 1, 'Color', 'k');
#			time3 = (timePlot(end) - outcplTime*1E6);
#			line([time3 time3],[-40 40], 'LineStyle', '--', 'LineWidth', 1, 'Color', 'k');
#			text(time1+0.2, 26, ['Incoupling \rightarrow'],'FontSize',11, 'HorizontalAlignment','right', ...
#		'FontWeight', 'bold', 'Color', 'k')
#			text(time3, -16, ['\leftarrow Outcoupling'],'FontSize',11, 'HorizontalAlignment','left', ...
#		'FontWeight', 'bold', 'Color', 'k')
#			text(time2+0.2, 12, ['Chirp end \rightarrow'],'FontSize',11, 'HorizontalAlignment','right', ...
#		'FontWeight', 'bold', 'Color', 'k')
#			axis([-2.2 timePlot(end)+2.4 -40 40])
#		  #set(gca, 'XTick', 0:5:floor(max(timePlot)), 'XMinorTick', 'on')
#		  
#			# write output to file
#			#wData = [timePlot; newPot];
#			#fid = fopen(['waveform' num2str(vInit) 'To' num2str(vFinal) '.txt'],'w');
#			#fprintf(fid, '#-10s  #-12s  #-12s  #-12s  #-12s  #-12s  #-12s\n', 'Time', 'Electrode 1', ...
#		#'Electrode 2', 'Electrode 3', 'Electrode 4', 'Electrode 5', 'Electrode 6');
#			#fprintf(fid,'#-.10f  #-+12.10f  #-+12.10f  #-+12.10f  #-+12.10f  #-+12.10f  #-+12.10f\n',wData);
#			#fclose(fid);
#	end

	# delay between Rydberg excitation and onset of waveform potentials
	PCBPotOffset = np.zeros((6, round(waveformStartDelay/timeStep)))
	newPot = np.concatenate((PCBPotOffset, newPot), 1)
	# if any entry is NaN replace with 0
	newPot[np.isnan(newPot)] = 0
	return newPot
