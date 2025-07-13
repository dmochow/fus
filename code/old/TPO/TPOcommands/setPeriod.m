function [out] = setPeriod(serialTPO,burstPeriod)
% setPeriod sets the burst period of the TPO (in microseconds) in 10 microsecond intervals
%   setPeriod(serialTPO,burstPeriod)
Period = round(burstPeriod,-1);
outStr = ['PERIOD=' num2str(Period)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
