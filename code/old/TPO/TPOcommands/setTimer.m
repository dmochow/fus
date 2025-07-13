function [out] = setTimer(serialTPO, timerSeconds)
% setTimer sets the treatment timer in milliseconds.
%   setTimer(serialTPO, timerSeconds)
%   Returns 0 if operation is succesfull

timerSeconds = round(timerSeconds);
outStr = ['TIMER=' num2str(timerSeconds)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
