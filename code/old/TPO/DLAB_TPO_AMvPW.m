%% TPO script to call TPO and set static functions with rampMode

%%
addpath('TPOcommands') % adds functions for issuing TPO commands to workspace

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

clear all; close all; clc;

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

%% Set TPO Power
%%%%%%%%%%%%%%%%%%%%%%% **** SET POWER HERE!!! ***** %%%%%%%%%%%%%%%%%%%%%%

setTPOPower = 21;    % in mW low: 21; high: 94
setType = 'AM'; %AM or PW

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Physical constants
soundSpeed = 1485;              % in meters per second

%% Static Parameters
xdrCenterFreq = 2.5E6;            % in hertz

nSystemChannels = 2;
tpoPower        = setTPOPower;    % in mW %low CW: 9; high CW: 51; low AM: 21; high AM: 94
tpoTimer        = 1000*60*3*1000;     % in microseconds
focus           = 12500;      % in micrometers

% set AM Ramping if necessary
if strcmp(setType,'AM')
    tpoBurstLength  = 25000;     % in microseconds
    tpoBurstPeriod  = 25000;    % in microseconds
    tpoRampMode     = 2;         % 0 = off, 1 = linear, 2 = tukey
    tpoRampLength   = 12500;       % in microseconds
elseif strcmp(setType,'PW')
    tpoBurstLength  = 12500;     % in microseconds
    tpoBurstPeriod  = 25000;    % in microseconds
    tpoRampMode     = 0;         % 0 = off, 1 = linear, 2 = tukey
end

%% setting all of the static parameters
setLocal(serialTPO,0);

setGlobalFreq(serialTPO,  xdrCenterFreq);
setGlobalPower(serialTPO,  tpoPower); % always set power after frequency or you may limit TPO

setFocus(       serialTPO, focus);
setBurst(       serialTPO,  tpoBurstLength);
setPeriod(      serialTPO,  tpoBurstPeriod);
setTimer(       serialTPO,  tpoTimer);
setRampMode(    serialTPO, tpoRampMode);

if tpoRampMode==0
else
    setRampLength(  serialTPO,  tpoRampLength);
end

%%
% startTPO(serialTPO);
% stopTPO(serialTPO);