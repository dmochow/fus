function [out] = setPhase(serialTPO, channel, phase)
% setPhase set the phase of a particular TPO channel in radians
%   setPhase(serialTPO, channel, Theta) sets phase angle of Chan
while phase < 0
    phase = phase + 2*pi;
end

while phase > 2*pi
    phase = phase - 2*pi;
end

phaseReg = round(phase*3600/(2*pi), 0);

outStr = ['PHASE' num2str(channel) '=' num2str(phaseReg)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
