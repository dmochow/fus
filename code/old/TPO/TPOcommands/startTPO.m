function [out] = startTPO(serialTPO)
% startTPO(serialTPO) initiates TPO output
outStr = 'START';
fprintf(serialTPO,outStr);
reply = fscanf(serialTPO);
disp(reply);
end
