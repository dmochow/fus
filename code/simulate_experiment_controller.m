clear all; close all; clc

% TR 
TR = 1.0; % units of seconds
% The sonication_duration should be programmed to have a duration that is a
% multiple of the TR duration (i.e., sonication duration = 1 TR)

% experiment parameters
n_sonications_todo = 10;
n_TRs_between_sonications = 3; % units of TRs

% initialize program counters
n_TRs_received = 0;
n_sonications_done = 0;

% 
event_types={};
event_times={};

while 1

    % simulate TRs coming in 
    pause(TR) % in your script, this needs be replaced with waiting until a "5" is received

    n_TRs_received = n_TRs_received + 1;
    timestamp = datetime;
    timestamp.Format='dd-MMM-uuuu HH:mm:ss:ms';
    str_ts = string(timestamp);
    sprintf("Received TR %d%n", n_TRs_received)
    event_types=cat(1,event_types,'TR');
    event_times=cat(1,event_times,str_ts);

    if rem(n_TRs_received, n_TRs_between_sonications) == 0
        timestamp = datetime;
        timestamp.Format='dd-MMM-uuuu HH:mm:ss:ms';
        str_ts = string(timestamp);
        sprintf("Starting sonication %d%n", n_sonications_done+1)
        n_sonications_done = n_sonications_done + 1;
        event_types=cat(1,event_types,'sonication');
        event_times=cat(1,event_times,str_ts);

        if n_sonications_done == n_sonications_todo
            break
        end

    end


end


tbl = table(event_types, event_times);
writetable(tbl,"event_table_sample.txt")
