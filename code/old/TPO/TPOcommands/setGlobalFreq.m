function [out] = setGlobalFreq(serialTPO, frequency)
% setFreq sets the frequency (in Hz) of a particular TPO channel in kHz
%   setFreq(serialTPO, channel, frequency)
frequency = round(frequency);
outStr = ['GLOBALFREQ=' num2str(frequency)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
