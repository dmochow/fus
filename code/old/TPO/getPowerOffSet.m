function [out] = getPowerOffSet(serialTPO, channel)
% getPowerOffSet will query the TPO for the power off set for a particular
% channel
%   getPowerOffSet(serialTPO, channel)

outStr = ['ADJPOWEROFS' num2str(channel) '?'];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
