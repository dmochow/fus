%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Annular Array demo script
%      
%       This script is provided with your TPO to demonstrate programmatic
%       control with an annular array transducer. The script is designed to
%       work with any four channel transducer with the appropriate
%       transducer parameters entered. The phase of each channel is
%       adjusted to move the pressure maxima away from its natural focus
%       orthogonally to the exit plane.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


%% Physical constants
soundSpeed = 1485;              % in meters per second

%% Transducer parameters

xdrCenterFreq = 2.5E6;            % in hertz

curvatureRadius = 20.0;        % in mm

el1InnerRad = 25.4*0.632;       % in mm
el1OuterRad = 25.4*0.818;     	% in mm

el2InnerRad = 25.4*0.828;       % in mm
el2OuterRad = 25.4*0.974;       % in mm


%% XDR parameter computation

elElevation = @(x) -sqrt(curvatureRadius.^2 - x.^2) + curvatureRadius;

el1AvgRad = (el1InnerRad + el1OuterRad)/2;
el2AvgRad = (el2InnerRad + el2OuterRad)/2;

el1Elev = elElevation(el1AvgRad);
el2Elev = elElevation(el2AvgRad);

%% phase offset function in radians

elPhaseCalc = @(x, y, F) xdrCenterFreq*1000*2*pi*(sqrt(x.^2 + (F - y).^2) - F)/(soundSpeed*1000);

el1Phase = elPhaseCalc(el1AvgRad, el1Elev, curvatureRadius);
el2Phase = elPhaseCalc(el2AvgRad, el2Elev, curvatureRadius);

%% system parameters
nSystemChannels = 2;
tpoPower        = 5000;    % in mW
tpoBurstLength  = 25000;     % in microseconds
tpoBurstPeriod  = 25000;    % in microseconds
tpoTimer        = 180000000;     % in microseconds

tpoRampMode_off = 1;         % 0 is the default value and means no ramping
tpoRampMode     = 2;         % Tukey
tpoRampLength   = 12500;       % in microseconds

%% setting all of the static parameters
setLocal(serialTPO,0);

for i = 1:nSystemChannels
    setFreq(    serialTPO,  i,  xdrCenterFreq);
    setPower(   serialTPO,  i,  tpoPower);       % always set power after frequency or you may limit TPO
end

setBurst(       serialTPO,  tpoBurstLength);      
setPeriod(      serialTPO,  tpoBurstPeriod); 
setTimer(       serialTPO,  tpoTimer);  
setRampMode(    serialTPO, tpoRampMode);
setRampLength(  serialTPO,  tpoRampLength);




% %% Display calculated/dynamic parameters of burst
% 
% focusPos = 0:2:30;
% 
% for i = 1:length(focusPos)
%     
% el1Phase = elPhaseCalc(el1AvgRad, el1Elev, curvatureRadius + focusPos(i));
% el2Phase = elPhaseCalc(el2AvgRad, el2Elev, curvatureRadius + focusPos(i));
% 
% setPhase(serialTPO, 1, 0);
% setPhase(serialTPO, 2, el2Phase - el1Phase);
% 
% % Ensure that abort command follows enable command or the TPO can freeze
% % up!
% setRampMode(serialTPO, tpoRampMode);
% 
% startTPO(serialTPO);
% disp(['intended focus: ' num2str(curvatureRadius + focusPos(i))]);
% 
% pause(0.1);
% 
% stopTPO(serialTPO);
% 
% 
% end
% 
% 
% %% Incoherent mode
% 
% beatFreq = 1/(tpoBurstLength*1E-6); % Frequency separation in kHz
% 
% setFreq(serialTPO,1,round(xdrCenterFreq - 0.5*beatFreq,0));
% setFreq(serialTPO,2,round(xdrCenterFreq + 0.5*beatFreq,0));
% 
% setPhase(serialTPO, 1, 0);
% setPhase(serialTPO, 2, 0);
% 
% 
% startTPO(serialTPO);
% disp('incoherent mode');
% 
% pause(1.5);
% 
% stopTPO(serialTPO);
% 
% %% Sets TPO to local and closes COM port, deletes object
% 
% setLocal(serialTPO,1);
% 
% fclose(serialTPO);
% delete(serialTPO);
% 
