function [out] = setPower(serialTPO,channel, electricalWatts)
% setPower sets the power of the TPO in milliwatts.
%   setPower(serialTPO, electricalWatts)
electricalWatts = round(electricalWatts);
outStr = ['POWER' num2str(channel) '=' num2str(electricalWatts)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
