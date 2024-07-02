function val = offtangent(M, x, v)
% Quantifies how far off v is as a tangent vector at x on the manifold M.
%
% function val = offtangent(M, x, v)
%
% Given a manifold M (obtained from a Manopt factory), a point x on M, and
% a would-be tangent vector v at x, the output val is a real number which
% is zero (up to machine precision) if v is indeed a tangent vector at x.
%
% The larger val is, the further away v is from being tangent.
%
% If v does not even have the correct format (e.g., a matrix of the wrong
% size, or not a structure with the expected fields, etc.), then val = Inf.
%
% If the manifold M does not provide sufficient means to run the check,
% then val = NaN.
%
% To 'make' v tangent, you may be able to call M.tangent(x, v).
%
% This tool is mostly used for debugging. It is a wrapper around the
% manifold's own M.offtangent, with a fallback if that is not implemented.
% 
% See also: offmanifold

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 2, 2024.
% Contributors:
% Change log:

    try
        if isfield(M, 'offtangent')
            % First ask the manifold itself if it can do the check.
            val = M.offtangent(x, v);
        elseif all(isfield(M, {'tangent', 'lincomb', 'norm'}))
            vv = M.tangent(x, v);
            residual = M.lincomb(x, 1, v, -1, vv);
            val = M.norm(x, residual);
        else
            % Don't know how to run the check.
            val = NaN;
        end
    catch
        % An error was thrown, indicating v (or x) may not even have
        % the right format.
        val = Inf;
    end

end
