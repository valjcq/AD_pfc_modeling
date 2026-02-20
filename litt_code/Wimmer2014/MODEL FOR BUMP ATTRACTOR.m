%
%   FIRING RATE NETWORK MODEL FOR BUMP ATTRACTOR
%
%   Continuous ring connectivity
%

clear all

%%%%% PARAMETERS

N=512; 		% number of "neurons" in the rate model
npop=8; 	% number of cues presented

totalTime=4200;	% total time of the simulation in ms
dt=2; 		% integration step in ms

tauE=20;	% time constant of rate equation for excitatory neurons
tauI=10;	% time constant of rate equation for inhibitory neurons

kappa=1.5;	% parameter defining concentration of e-to-e connectivity
GEE=6;	 	% strength of excitation to excitatory neurons
GEI=4;		% strength of excitation to inhibitory neurons
GIE=3.4;	% strength of inhibition to excitatory neurons
GII=0.85;	% strength of inhibition to inhibitory neurons

I0E=0.2;	% external bias current to excitatory neurons
I0I=0.5;	% external bias current to inhibitory neurons

sigE=1;		% standard deviation of additive noise in rate equation of e-cells
sigI=3;		% standard deviation of additive noise in rate equation of i-cells

stimon = 1000;	% time when external stimulus is applied in ms
stimoff = 1500;	% time when external stimulus ceases in ms
stim = 200;  	% strength of external stimulus
delayend=3500;	% time when delay ends in ms, and external input is applied to erase memory

%%%%% PRELIMINARY CALCULATIONS

rE=zeros(N,1);
rI=zeros(N,1);
nsteps=floor(totalTime/dt);
delayPop=zeros(N,1);

% E-to-E connectivity
theta = [0:N-1]/N*2*pi;
v = exp(kappa*cos(theta));
v = v/sum(v);
WE = gallery('circul',v);

% stimulus parameters
theta=theta-pi;
v = exp(kappa*cos(theta));
v = v/sum(v);
stimulus = stim*v';
stimon = floor(stimon/dt);
stimoff = floor(stimoff/dt);
delayend = floor(delayend/dt);
delaywin = floor(100/dt); %100 ms window

% input-output function for all cells, as used previously (Brunel, Cereb Cortex 13:1151, 2003)
f = inline('x.*x.*(x>0).*(x<1)+sqrt(4*x-3).*(x>=1)');

% population vector decoder given the rates r for neurons with selectivity th
decode = inline('atan2(sum(r.*sin(th)),sum(r.*cos(th)))','r','th');

% prepare figure panels for graphical output
F=figure; 
set(F,'color','w')

H1=subplot(2,1,1); % upper left panel for e-cell activity
plot(([0:npop]-npop/2)/(npop/2)*pi,10.5,'.k','markersize',10,'color',0.8*[1 1 1],'clipping','off')
ylim([0 12])
xlim([-pi pi])
ylabel('e-cell rate')
set(H1,'xtick',[-2:2]/2*pi)
set(H1,'xticklabel',[-2:2]/2*180)
box off
hold on

H2=subplot(2,1,2); % lower panel for i-cell activity
ylabel('i-cell rate')
xlabel('neuron (deg)')
ylim([0 10])
xlim([-pi pi])
set(H2,'xtick',[-2:2]/2*pi)
set(H2,'xticklabel',[-2:2]/2*180)
hold on

nbl=floor(N/npop);

%%%% SIMULATION LOOP

for i=1:nsteps,

  % additive noise for each population
  noiseE = sigE*randn(N,1);
  noiseI = sigI*randn(N,1);
  
  % current input to each population
  IE=GEE*WE*rE+(I0E-GIE*mean(rI))*ones(N,1);
  II=(GEI*mean(rE)-GII*mean(rI)+I0I)*ones(N,1);
  
  % external task-dependent inputs
  if i>stimon & i<stimoff, 
    IE=IE+stimulus; % cue stimulus before delay
  end
  if i>delayend & i<delayend+(stimoff-stimon),
    IE=IE-stim;  % erasing global input after delay
  end

  if i>delayend-delaywin & i<=delayend,
    delayPop = delayPop + rE/delaywin;
  end

  % integration with time-step dt: Newton method
  rE = rE + (f(IE) - rE + noiseE)*dt/tauE;
  rI = rI + (f(II) - rI + noiseI)*dt/tauI;

  % get decoded angle from network activity  
  ang=decode(rE,theta');
  if i<delayend, response=ang; end

  % graphical output
  subplot(H1)

  if exist('HP1'), delete(HP1), end
  if i>stimoff & i<delayend,
    HP1=plot(theta,rE,'r',[0 ang],10.5*[1 1],'k-',ang,11,'kv','markersize',10,'markerfacecolor','k');
  else
    HP1=plot(theta,rE,'r',ang,11,'kv','markersize',10,'markerfacecolor','k');
  end
  if i>stimon & ~exist('HL'), 
    HL=line([-pi/4 pi/4],[0 0],'color','k','linewidth',4,'clipping','off');
  end
  if i>stimoff & exist('HL'), delete(HL), clear HL, end
  if i>delayend & ~exist('HL2'), 
    HL2=line([-pi pi],[0 0],'color','k','linewidth',4,'clipping','off');
  end
  if i>delayend+(stimoff-stimon) & exist('HL2'), delete(HL2), clear HL2, end

  subplot(H2)
  if exist('HP2'), delete(HP2), end
  HP2=plot(theta,rI,'b');
  
  drawnow

end


