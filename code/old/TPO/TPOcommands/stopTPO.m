function [out] = stopTPO(serialTPO)
% stopTPO(serialTPO) stops TPO output

outStr = 'ABORT';
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
