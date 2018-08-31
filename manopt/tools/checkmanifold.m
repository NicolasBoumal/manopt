function checkmanifold(M)
% Run a collection of tests on a manifold obtained from a manopt factory
% 
% function checkmanifold(M)
%
% M should be a manifold structure obtained from a Manopt factory. This
% tool runs a collection of tests on M to verify (to some extent) that M is
% indeed a valid description of a Riemannian manifold.
%
% This tool is work in progress: your suggestions for additional tests are
% welcome on our forum.
%
% See also: checkretraction

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Aug. 31, 2018.
% Contributors: 
% Change log: 

    assert(isstruct(M), 'M must be a structure.');
    
    %% List required fields that must be function handles here
    list_of_functions = {'name', 'dim', 'inner', 'norm', 'typicaldist', ...
                         'proj', 'tangent', 'egrad2rgrad', 'retr', ...
                         'rand', 'randvec', 'zerovec', 'lincomb'};
    for k = 1 : numel(list_of_functions)
        field = list_of_functions{k};
        if ~(isfield(M, field) && isa(M.(field), 'function_handle'))
            fprintf('M.%s must be a function handle.\n', field);
        end
    end
    
    %% List recommended fields that should be function handles here
    list_of_functions = {'dist', 'ehess2rhess', 'exp', 'log', 'hash', ...
                         'transp', 'pairmean', 'vec', 'mat', ...
                         'vecmatareisometries'};
    for k = 1 : numel(list_of_functions)
        field = list_of_functions{k};
        if ~(isfield(M, field) && isa(M.(field), 'function_handle'))
            fprintf('M.%s should be a function handle.\n', field);
        end
    end
    
    %% Checking exp and dist
    try
        x = M.rand();
        v = M.randvec(x);
        t = randn(1);
        y = M.exp(x, v, t);
        d = M.dist(x, y);
        fprintf('dist(x, M.exp(x, v, t)) - abs(t)*M.norm(x, v) = %g (should be zero)\n', d - abs(t)*M.norm(x, v));
    catch up %#ok<NASGU>
        fprintf('Couldn''t check exp and dist.\n');
        % Perhaps we want to rethrow(up) ?
    end

end

