function stop = stopinteractive(problem, x, info, last) %#ok<INUSL>
% Attempt at a stopping criterion based on: a figure open; and if the user
% closes it at any point, then the solver terminates. But it didn't work...
% NB, Aug. 2, 2018.

    persistent haxis;
    if isempty(haxis)
        haxis = gca();
        drawnow;
    end

%     if last == 1
%         figure(figureid);
%         drawnow;
%     end
    
    % Doesn't work because, while the figure does close, its handle remains
    % live until the computations are over..
    if size(findobj(haxis)) <= 0 % ~ishandle(haxis) %~(ishandle(figureid) && findobj(figureid, 'type', 'figure') == figureid)
        stop = true;
    else
        fprintf('\n\nAll good...\n\n');
        stop = false;
    end

end
