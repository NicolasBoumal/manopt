function [version, released] = manopt_version()
% Returns the version of the Manopt package you are running, as a vector.
%
% function [version, released] = manopt_version()
%
% version(1) is the primary version number.
% released is the date this version was released, in the same format as the
% string(datetime('today')) function in Matlab.

    version = [8, 0, 0];
    released = '05-Jul-2024';

end
