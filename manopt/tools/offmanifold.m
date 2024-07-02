function val = offmanifold(M, x)
% Quantifies how far off x is as a point on the manifold M.
%
% function val = offmanifold(M, x)
%
% Given a manifold M (obtained from a Manopt factory) and a would-be
% point x on M, the output val is a real number which is zero (up to
% machine precision) if x is indeed a point on M.
%
% The larger val is, the further away x is from being on the manifold.
%
% If x does not even have the correct format (e.g., a matrix of the wrong
% size, or not a structure with the expected fields, etc.), then val = Inf.
%
% If the manifold M does not provide sufficient means to run the check,
% then val = NaN.
%
% This tool is mostly used for debugging. It is a wrapper around the
% manifold's own M.offmanifold. Fallbacks could be implemented here.
% 
% See also: offtangent

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 2, 2024.
% Contributors:
% Change log:

    try
        if isfield(M, 'offmanifold')
            % First ask the manifold itself if it can do the check.
            val = M.offmanifold(x);
        else
            % If not, don't know how to run the check.
            val = NaN;
        end
    catch
        % An error was thrown, indicating x may not even have
        % the right format.
        val = Inf;
    end

end
