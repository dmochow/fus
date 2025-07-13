function [out] = setRampLength(serialTPO, rampLength)
% setTimer specifies the time duration over which ramping occurs 
%   setTimer(serialTPO, timerSeconds)
%   Returns 0 if operation is succesfull
rampLength = round(rampLength);
outStr = ['RAMPLENGTH=' num2str(rampLength)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
