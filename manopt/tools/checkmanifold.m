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
% welcome on our forum or as pull requests on GitHub.
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
            fprintf(['M.%s should ideally (but does not have to) be ' ...
                     'a function handle.\n'], field);
        end
    end
    
    %% Check random generators
    try
        x = M.rand();
        v = M.randvec(x);
        fprintf('Random tangent vector norm: %g (should be 1).\n', ...
                M.norm(x, v));
    catch up %#ok<NASGU>
        fprintf('Couldn''t check rand, randvec.\n');
    end
    
    %% Checking exp and dist
    try
        x = M.rand();
        v = M.randvec(x);
        t = randn(1);
        y = M.exp(x, v, t);
        d = M.dist(x, y);
        fprintf('dist(x, M.exp(x, v, t)) - abs(t)*M.norm(x, v) = %g (should be zero).\n', d - abs(t)*M.norm(x, v));
    catch up %#ok<NASGU>
        fprintf('Couldn''t check exp and dist.\n');
        % Perhaps we want to rethrow(up) ?
        % Alternatively, we could check if exp and dist are available and
        % silently pass this test if not, but this way is more informative.
    end
    
    %% Checking mat, vec, vecmatareisometries
    try
        x = M.rand();
        u = M.randvec(x);
        v = M.randvec(x);
        U = M.vec(x, u);
        V = M.vec(x, v);
        if ~iscolumn(U) || ~iscolumn(V)
            fprintf('M.vec should return column vectors: they are not.\n');
        end
        if ~isreal(U) || ~isreal(V)
            fprintf('M.vec should return real vectors: they are not real.\n');
        end
        fprintf('M.vec seems to return real column vectors, as intended.\n');
        ru = M.norm(x, M.lincomb(x, 1, M.mat(x, U), -1, u));
        rv = M.norm(x, M.lincomb(x, 1, M.mat(x, V), -1, v));
        fprintf(['Checking mat/vec are inverse pairs: ' ...
                 '%g, %g (should be two zeros).\n'], ru, rv);
        a = randn(1);
        b = randn(1);
        fprintf('Checking if vec is linear: %g (should be zero).\n', ...
                norm(M.vec(x, M.lincomb(x, a, u, b, v)) - (a*U + b*V)));
        if M.vecmatareisometries()
            fprintf('M.vecmatareisometries says true.\n');
        else
            fprintf('M.vecmatareisometries says false.\n');
        end
        fprintf('If true, this should be zero: %g.\n', ...
                    U(:).'*V(:) - M.inner(x, u, v));
    catch up %#ok<NASGU>
        fprintf('Couldn''t check mat, vec, vecmatareisometries.\n');
    end

end

