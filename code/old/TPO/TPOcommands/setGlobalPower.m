function [out] = setGlobalPower(serialTPO, electricalWatts)
% setPower sets the power of the TPO in milliwatts.
%   setPower(serialTPO, electricalWatts)
electricalWatts = round(electricalWatts);
outStr = ['GLOBALPOWER=' num2str(electricalWatts)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
