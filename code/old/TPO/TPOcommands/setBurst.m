function [out] = setBurst(serialTPO, microseconds)
% setBurst sets the burst length of the TPO (in microseconds) in 10 microsecond intervals
%   setBurst(serialTPO, burstMicroseconds)
microseconds = round(microseconds, -1);
outStr = ['BURST=' num2str(microseconds)];
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
