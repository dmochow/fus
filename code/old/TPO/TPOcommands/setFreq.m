function [out] = setFreq(serialTPO, channel, frequency)
% setFreq sets the frequency (in Hz) of a particular TPO channel in kHz
%   setFreq(serialTPO, channel, frequency)
frequency = round(frequency);
outStr = ['FREQ' num2str(channel) '=' num2str(frequency)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
