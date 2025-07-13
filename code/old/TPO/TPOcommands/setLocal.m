function [out] = setLocal(serialTPO,local)
% setFreq sets the TPO into either local mode(1) or remote(0)
%   setLocal(serialTPO, local)
    outStr = ['LOCAL=' num2str(local)];
    fprintf(serialTPO,outStr);
    reply = fscanf(serialTPO);
    disp(reply);
end
