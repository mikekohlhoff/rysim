function GasExpansion(seedingGas)
    
    % generate initial distribution from expanded valve cloud
    % up to point of laser excitation

    % CONSTANTS
    % mass of hydrogen
    mass = 1.6735327160314e-27;
    % Boltzmann constant / J/K
    kB = 1.380648813e-23;
    % Bohr radius / m
    a0 = 5.291772109217e-11; 
    % elementary charge / C
    e = 1.60217656535e-19;


    % COMPONENT DIMENSIONS and TIMINGS
    % distance capillary to intersection atom beam/laser intersection
    % value from Erik's program - (shift assumed midpoint of chamber to new
    % intersection when different ports used) 
    % excitation in capillary to Rydberg excitation
    totalDist = (46.7 - 3.54)*1E-2;
    lengthCapillary = 14*1E-3;
    % midpoint capillary to skimmer
    skimmerDist = lengthCapillary + 3*1E-2;

    radiusCapillary = 0.5*1E-3;
    radiusSkimmer = 0.5*1E-3;


    % GAS BEAM parameters upon leaving pulse valve
    % cf. excimer beam width
    atomBeamLength = 3e-3;
    atomBeamWidth = 2*radiusCapillary;

    % Kr/Xe possible, V = 440/330
    if strcmp(seedingGas, 'NH3')
    vBeamFwd = 2500;
    elseif strcmp(seedingGas, 'He')
    vBeamFwd = 1350;
    elseif strcmp(seedingGas, 'Ne')
    vBeamFwd = 975;
    elseif strcmp(seedingGas, 'Ar')
    vBeamFwd = 700;
    end

    % mass for velocity for Maxwell-Boltzmann distribution
    if strcmp(seedingGas, 'NH3')
        massMolBeam = mass*1;
    elseif strcmp(seedingGas, 'He')
        massMolBeam = mass*4;
    elseif strcmp(seedingGas,  'Ne')
        massMolBeam = mass*20;
    elseif strcmp(seedingGas,  'Ar')
        massMolBeam = mass*40;
    end


    % LASER PARAMETERS
    % delay to intercept maximum of velocity distribution, 180 = delay(2500ms-1)
    % measured 5mm widht before f=200mm lens
    excimerWidth = 2E-3;
    laserDelay = totalDist/vBeamFwd;
    % laser radii focussed down
    radiusRydbergLaser = 0.5*1E-3;


    % NUMBER of particles at point of Hydrogen dissociation
    % typical density of 1E13-12/cm3 after expansion
    initDensity = .7E11*1E6;
    excimerVolume = pi*radiusCapillary^2*excimerWidth;
    NCapil = round(initDensity*excimerVolume);
    display(['Initial number of particles in capillary laser volume: ', num2str(NCapil, '%.2e')]) 
    fprintf('\n')
    if NCapil > 10E8
        display('Error: Number of particles too high for creating arrays')
        return
    end

    
    % STARK STATE
    n = 30;
    % maximally blue shifted state, assumed m_l = 0
    k = n - 1;


    % set INITIAL CONDITIONS for position and velocity Monte Carlo
    % get starting DISTRIBUTION from GAS EXPANSION from capillary
    % initial position at middle of capillary 
    r0 = [0;0;0]; 
    v0 = [0;0;vBeamFwd];

    % intial POSITIONS and spread (if not uniform distributed)
    % uncertainty in position due to 10ns excimer paser pulse not included
    rSigma = [radiusCapillary; radiusCapillary; excimerWidth/2];

    % flat (uniform) distribution within capillary along disk
    rRadiusInit = sqrt(unifrnd(0, radiusCapillary, 1, NCapil))*sqrt(radiusCapillary);
    rPhiInit = unifrnd(0, 2*pi, 1, NCapil);

    % transform to polar coordinates
    [xCart, yCart] = pol2cart(rPhiInit, rRadiusInit);

    rx = r0(1) + xCart;
    ry = r0(2) + yCart;

    % uniformly distributed along height of cylinder, excimer laser profile
    % assumed to have rectangularly shaped output
    rz = r0(3) + unifrnd(lengthCapillary/2 - excimerWidth/2, ...
                         lengthCapillary/2 + excimerWidth/2, 1, NCapil);

    %% normal distributed around median r0 mu+sigma*randn(noParticles,1)
    %rx(1,:) = r0(1) + rSigma(1).*randn(N,1); 
    %ry(1,:) = r0(2) + rSigma(2).*randn(N,1); 
    %rz(1,:) = r0(3) + rSigma(3).*randn(N,1);

    rInit = [rx; ry; rz];

    % capillary size discrimination, will only affect normal distribution
    rOutCapillary = sqrt(rInit(1,:).^2 + rInit(2,:).^2);
    rInit = rInit(:,rOutCapillary < radiusCapillary);
    NInit = length(rInit(3,:));

    % VELOCITY SPREAD, unseeded 80K, He 20K, Ne 30K, Ar 40K 
    % initial velocity /mm with forward expansion and transverse spread sigma_x,y
    % taken from thesis Eric So (compare to spread in Hogan et al., ~n*10mK
    if strcmp(seedingGas, 'NH3')
        temperature = 80;
    elseif strcmp(seedingGas, 'He')
        temperature = 22;
    elseif strcmp(seedingGas,  'Ne')
        temperature = 30;
    elseif strcmp(seedingGas,  'Ar')
        temperature = 1;
    end

    % spread for one degree of freedom 
    vSigma = sqrt(kB*temperature/massMolBeam)*[1;1;1];
    % normal distributed gas package around median velocity
    vx = v0(1) + vSigma(1).*randn(1, NInit); 
    vy = v0(2) + vSigma(2).*randn(1, NInit); 
    vz = v0(3) + vSigma(3).*randn(1, NInit);
    vInit = [vx; vy; vz];


    % SKIMMER size discrimination
    timeSkimmer = (skimmerDist - rInit(3,:))./vInit(3,:);
    % all particles propagated to skimmer, then discriminated
    rSkimmer = rInit + bsxfun(@times,vInit,timeSkimmer);

    rOutSkimmer = sqrt(rSkimmer(1,:).^2 + rSkimmer(2,:).^2);
    indexOutSkimmer = rOutSkimmer <= radiusSkimmer;

    rSkimmer = rInit(:,indexOutSkimmer);
    vSkimmer = vInit(:,indexOutSkimmer);


    % PROPAGATION to intersection of atom beam and lasers
    rLasersIntercept = rSkimmer + vSkimmer.*laserDelay;
    % center of gas packet at origin for plotting
    rLasersIntercept = rLasersIntercept - repmat([0; 0; (vBeamFwd*laserDelay + ...
    .5*lengthCapillary)], 1, length(rLasersIntercept));

    % R_y(angle)
    rotPhi = @(alpha) [cosd(alpha) 0 sind(alpha); 0 1 0; -sind(alpha) 0 cosd(alpha)];
    % R_x(angle)
    rotTheta = @(alpha) [1 0 0; 0 cosd(alpha) -sind(alpha); 0 sind(alpha) cosd(alpha)];

    % rotate to laser frame
    phi = 20;
    theta = 45;
    % rotation counter-clockwise at 45 degrees
    rLasersIntercept = rotTheta(theta)*rLasersIntercept;

    % laser intersection volume discrimination
    rOutLasersIntercept = sqrt(rLasersIntercept(1,:).^2 + rLasersIntercept(2,:).^2);
    indexOutLasers = rOutLasersIntercept <= radiusRydbergLaser;

    rLasersIntercept = rLasersIntercept(:,indexOutLasers);
    vLasersIntercept = vSkimmer(:,indexOutLasers);

    % rotate back to atom beam frame clockwise
    rLasersIntercept = rotTheta(-theta)*rLasersIntercept;

    % match geometry in Simion potential arrays 
    rSwap = rLasersIntercept(1:2,:);
    rLasersIntercept = [flipud(rSwap); rLasersIntercept(3,:)];

    vSwap = vLasersIntercept(1:2,:);
    vLasersIntercept = [flipud(vSwap); vLasersIntercept(3,:)];

    display(['Number of particles at laser excitation: ', num2str(length(rLasersIntercept), '%.2e'), ...
        ' (= ', num2str(length(rLasersIntercept)*100/NCapil, 1), '% of initial N)'])

    clearvars r* -except rLasersIntercept
    clearvars v* -except vLasersIntercept vBeamFwd
    clearvars index*

    clear xCart yCart timeSkimmer

    % at tranverse T as in thesis E.So particle density at point
    % of excitation is low (<1E3); fitting procedure for more particles
    x = rLasersIntercept(1,:)';
    y = rLasersIntercept(2,:)';
    z = rLasersIntercept(3,:)';

    vx = vLasersIntercept(1,:)';
    vy = vLasersIntercept(2,:)';
    vz = vLasersIntercept(3,:)';

    clear *LasersIntercept

    % Transform the data to the copula scale (unit square) using a 
    % kernel estimator of the cumulative distribution function.
    u = ksdensity(x,x,'function','cdf');
    v = ksdensity(y,y,'function','cdf');
    w = ksdensity(z,z,'function', 'cdf');

    vu = ksdensity(vx, vx, 'function', 'cdf');
    vv = ksdensity(vy, vy, 'function', 'cdf');
    vw = ksdensity(vz, vz, 'function', 'cdf');

    % fit t copula (linearly correlated)
    % ApproximateML vs ML (default)
    method = 'ML';
    [Rho,nu] = copulafit('t',[u v w vu vv vw],'Method',method);

    save(['LaserExcitedDistribution' seedingGas '.mat'], 'x', 'y','z', 'vx', 'vy', 'vz', 'Rho', 'nu');
