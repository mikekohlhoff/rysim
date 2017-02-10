clear all
format compact

% PROGRAM INTERNAL MODES
% plotting 
plotModeResults = false;
% create new random sample every run or load old one

SeedingGases = {'NH3' 'He' 'Ne' 'Ar'};
SeedingGas = SeedingGases{2};

fprintf('\n')
display('Simulation started')
display('---------------------------------------------------------------------------------------')

% CONSTANTS
% mass of hydrogen
mass = 1.6735327160314e-27;
% Boltzmann constant / J/K
kB = 1.380648813e-23;
% Bohr radius / m
a0 = 5.291772109217e-11; 
% elementary charge / C
e = 1.60217656535e-19;

% Kr/Xe possible, v = 440/330
if strcmp(SeedingGas, 'NH3')
vBeamFwd = 2500;
elseif strcmp(SeedingGas, 'He')
vBeamFwd = 1350;
elseif strcmp(SeedingGas, 'Ne')
vBeamFwd = 975;
elseif strcmp(SeedingGas, 'Ar')
vBeamFwd = 700;
end


% STARK STATE
n = 30;
% maximally blue shifted state, assumed m_l = 0
k = n - 1;

% generate atom cloud from valve expansion if file does not exist
gasFile = ['LaserExcitedDistribution' SeedingGas];
if exist([gasFile '.mat'], 'file') == 2
    load(gasFile);
else
    GasExpansion(SeedingGas);
    load(gasFile);
end

% width of slice out of 3D distribution
xWidth = 15E-3;
% create roughly 1E4 particles for 2D slice from initial 3D expansion
if exist(['LaserExcitedDistribution2D' SeedingGas '.mat'], 'file') == 0
    SeedingGas
    % gives roughly 10E3 particles 
    NSim = 11.3E4;

    % build 2D distribution from 3D distribution
    r = copularnd('t',Rho,nu, NSim);
    u1 = r(:,1);
    v1 = r(:,2);
    w1 = r(:,3);
    vu1 = r(:,4);
    vv1 = r(:,5);
    vw1 = r(:,6);

    % Transform the random sample back to the original scale of the data
    xn = ksdensity(x,u1,'function','icdf');
    yn = ksdensity(y,v1,'function','icdf');
    zn = ksdensity(z,w1,'function','icdf');
    vxn = ksdensity(vx,vu1,'function','icdf');
    vyn = ksdensity(vy,vv1,'function','icdf');
    vzn = ksdensity(vz,vw1,'function','icdf');

    rLasersIntercept = zeros(3,NSim);
    vLasersIntercept = zeros(3, NSim);

    rLasersIntercept(1,:) = xn;
    rLasersIntercept(2,:) = yn;
    rLasersIntercept(3,:) = zn;
    vLasersIntercept(1,:) = vxn;
    vLasersIntercept(2,:) = vyn;
    vLasersIntercept(3,:) = vzn;

    save(['LaserExcitedDistribution2D' SeedingGas '.mat'],'rLasersIntercept','vLasersIntercept');
else
    load(['LaserExcitedDistribution2D' SeedingGas])
end

% choose particles within x-width
IndicesXWidth = find(rLasersIntercept(1,:) < -xWidth/2);
IndicesXWidth2 = find(rLasersIntercept(1,:) > xWidth/2);
IndicesXWidth = [IndicesXWidth IndicesXWidth2];
rLasersIntercept(1,:) = [];
vLasersIntercept(1,:) = [];
rLasersIntercept(:,IndicesXWidth) = [];
vLasersIntercept(:,IndicesXWidth) = [];

% match 2D geometry from simion, swap z to x direction
rLasersIntercept([1 2],:) = rLasersIntercept([2 1],:);
vLasersIntercept([1 2],:) = vLasersIntercept([2 1],:);

fprintf('\n')
display(['Initial distribution created, ' num2str(length(rLasersIntercept(2,:))) ' particles'])
display('---------------------------------------------------------------------------------------')

