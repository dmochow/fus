%% Set up trial definitions and parameters
clear all; close all; clc;

interSec=[6 11];
interStim=ones(750,1);
interStim(2:2:750)=2;

% Skull Attenuated Actual intensities: 13, 52, 208, 832, 3328
int(1,:) = [27 92 250 565 1196]; %AM
int(2,:) = [11 52 157 380 826]; %CW
int(3,:) = [27 92 250 565 1196]; %PW
rampMode = [2 0 0];

intName = {'0.013W', '0.052W', '0.208W', '0.832W', '3.328W'};

burstLen = [(1/300)*1000*1000 (1/300)*1000*1000 (1/300)*1000*1000*1/2];

%set up the nominal trials numbers
nTrials=750;

p=0;
wNames={'AM','CW','PW'};
stimOrder=cell(nTrials,1);
tpoPower=ones(750,1);
tpoBurstLength=ones(750,1);
tpoRampMode=ones(750,1);

for t=1:50
    for i=1:5
        for w=1:3            
            p=p+1;
            stimOrder{p}=[wNames{w} '_' intName{i}];
            tpoPower(p)=int(w,i); 
            tpoBurstLength(p)=burstLen(w); 
            tpoRampMode(p)=rampMode(w);
        end
    end
end

%% Set common parameters:
soundSpeed      = 1485;         % in meters per second
nSystemChannels = 2;            % 2 channels
xdrCenterFreq   = 2.5E6;        % in hertz
focus           = 12500;        % in micrometers
tpoTimer        = 1000*1000;    % in microseconds - 1 second stimulation
tpoBurstPeriod  = (1/300)*1000*1000;        % in microseconds - 1s periods
tpoRampLength   = (1/300)*1000*1000*1/2;        % in microseconds - 500 ms ramp for AM 

%%
addpath('D:\Desktop\TPO\TPOcommands') % adds functions for issuing TPO commands to workspace
outDir=('D:\Desktop\study8');

%% TPO control begins

% Code to clear com port if code was aborted
if exist('serialTPO', 'var')
    try
        fclose(serialTPO);
        delete(serialTPO);
    catch
        delete(serialTPO);
    end
end

newobjs = instrfind;
if ~isempty(newobjs)
    fclose(newobjs);
end

%% Finds available port and sets up serial object
% NOTE this is untested in environments outside of windows
try
    COMports = comPortSniff; % cell containing string identifier of com port
catch
    error('No COM ports found, please check TPO');
end
% Removes any empty cells
COMports = COMports(~cellfun('isempty',COMports));
len = length(COMports(:));
COMports = reshape(COMports,[len/2 2]);

%% Picks out TPO port
tempInd = strfind(COMports(:,1), 'Arduino Due');
indTPO = find(not(cellfun('isempty', tempInd)));

if isempty(indTPO)
    error( 'No TPO detected, please check your USB and power connections')
end
indTPO = indTPO(1); %Trims off multiple matches and takes first

%% Opens COM port to TPO
disp(['COM port: ' num2str(indTPO) '-' COMports{indTPO,1}]);
serialTPO = serial(['COM' num2str(COMports{indTPO,2})],'BaudRate', 115200,'DataBits', 8, 'Terminator', 'CR');
fopen(serialTPO);
pause(4); % 4 second pause to wait for power-on reset
reply = fscanf(serialTPO); % Dummy read to get tpo name and software version
disp(reply)

%%
setLocal(serialTPO,0);
setGlobalFreq(serialTPO,  xdrCenterFreq);

for i = 1:nSystemChannels
    setFreq(    serialTPO,  i,  xdrCenterFreq);
end

setFocus(       serialTPO, focus);
setBurst(       serialTPO,  tpoBurstPeriod);
setPeriod(      serialTPO,  tpoBurstPeriod);
setTimer(       serialTPO,  tpoTimer);

%% Defining and labeling
rat='r116';
cond='HPC_300Hz';
nTrials=750;
startPause=10;
ratDir=[outDir filesep rat];

%% Randomizing stimulation orders
rng('shuffle');

for r=1:round(rand*100)
    z=randperm(nTrials);
end

%% Run the automatic stimulation
%    setGlobalPower(serialTPO,  tpoPower(z(i))); % always set power after
%    frequency or you may limit TPO %This call doesn't work so you have to
%    set it per channel
clc;
randOrder=cell(nTrials,1);
interStimOrder=ones(nTrials,1);

fprintf('\n\n%s: %s is about to start\n\n', rat, cond);
pause(startPause);
for i=1:nTrials
    fprintf('\n\nStim #%d: %s; TPO Power = %d\n',i,stimOrder{z(i)},tpoPower(z(i)));
    randOrder{i}=stimOrder{z(i)};
    setPower(   serialTPO,  1,  tpoPower(z(i)));       % set the channel 1 power
    setPower(   serialTPO,  2,  tpoPower(z(i)));       % set the channel 2 power
    setBurst(       serialTPO,  tpoBurstLength(z(i)));
    setRampMode(    serialTPO, tpoRampMode(z(i)));
    if tpoRampMode(z(i))==2
        setRampLength(  serialTPO,  tpoRampLength);
    end
    startTPO(serialTPO);
    pause(interSec(interStim(z(i))));
    interStimOrder(i)=interSec(interStim(z(i)))-1;
end

fprintf('\n\nFinished %s %s.\n',rat,cond);
fprintf('Stop recording.\n\n');
fprintf('\n\nSaving out randOrder data.\n\n');

try
    cd(ratDir);
catch
    mkdir(ratDir);
    cd(ratDir);
end

save([rat '_' cond '.mat'], 'randOrder','interStimOrder');
fprintf('\n\nSaving completed.\n\n');


