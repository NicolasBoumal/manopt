function [stop, reason] = stopinteractive(problem, x, info, last) %#ok<INUSL>
% This became a tool: see stopifclosedfigure

    reason = 'Interactive stopping criterion: figure closed.';

    % Fix a likely unique figure id.
    figureid = 1465489213;
    
    % If first iteration, create a figure to capture interaction.
    if last == 1
        figure(figureid);
        text(0, 0, 'Close me to stop the Manopt solver.', 'FontSize', 16);
        axis tight;
        axis off;
        set(gcf, 'color', 'w');
        drawnow();
    end
    
    % Call the drawnow() ensures that, if the user closed the figure, then
    % that information will have been refreshed. This may create small
    % delays, but on the other hand interactive stopping criteria are
    % mostly useful for costly problems where this overhead should be
    % marginal.
    drawnow();
    if ~ishandle(figureid)      % If the figure was closed, stop.
        stop = true;
    else
        stop = false;
    end

end
