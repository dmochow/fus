function [out] = setFocus(serialTPO,focus)
% setFocus sets the desired focal depth of the transducer in 1 µm steps.
%   This is available for the NeuroFus series
%   setLocal(serialTPO, local)
    outStr = ['FOCUS=' num2str(focus)];
    fprintf(serialTPO,outStr);
    reply = fscanf(serialTPO);
    disp(reply);
end
