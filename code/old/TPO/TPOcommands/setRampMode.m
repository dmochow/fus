function [out] = setRampMode(serialTPO, modeArg)
%setRampMode Sets the TPO ramp aat the beginning and end of each burst
%   This is currently a BETA feature and requires a detailed understanding
%   of how it behaves. MAKE SURE TO READ THE MANUAL BEFORE USING. 
%   modeArg | Ramping function
%   0         No Ramping (inactive)
%   1         Linear
%   2         Tukey
%   3         Logarithmic
%   4         Exponential
modeArg = round(modeArg);
outStr = ['RAMPMODE=' num2str(modeArg)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end

