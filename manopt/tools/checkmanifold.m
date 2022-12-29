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
%   April 12, 2020 (NB):
%       Now checking M.dist(x, M.exp(x, v, t)) for several values of t
%       because this test is only valid for norm(x, tv) <= inj(x).
%   May 19, 2020 (NB):
%       Now checking M.dim().
%   Jan  8, 2021 (NB):
%       Added partial checks of M.inner, M.tangent2ambient, M.proj, ...

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
        z = M.lincomb(x, 1, v, -1, v);
        fprintf('norm(v - v)_x = %g (should be 0).\n', M.norm(x, z));
    catch up %#ok<NASGU>
        fprintf('Couldn''t check rand, randvec, lincomb.\n');
    end
    
    %% Check inner product
    try
        x = M.rand();
        
        % Check symmetry
        u = M.randvec(x);
        v = M.randvec(x);
        uv = M.inner(x, u, v);
        vu = M.inner(x, v, u);
        fprintf('<u, v>_x = %g, <v, u>_x = %g, difference = %g (should be 0).\n', uv, vu, uv-vu);
        
        % Check linearity (and owing to symmetry: bilinearity)
        a = randn();
        b = randn();
        w = M.lincomb(x, a, u, b, v);
        z = M.randvec(x);
        wz = M.inner(x, w, z);
        wzbis = a*M.inner(x, u, z) + b*M.inner(x, v, z);
        fprintf('<au+bv, z>_x = %g, a<u, z>_x + b<v, z>_x = %g, difference = %g (should be 0).\n', wz, wzbis, wz-wzbis);
        
        % Should check positive definiteness too: it's somehow part of the
        % check for M.dim() below.
        
    catch up %#ok<NASGU>
        fprintf('Couldn''t check inner.\n');
    end
    
    %% Check tangent2ambient, proj, norm
    try
        x = M.rand();
        v = M.randvec(x);
        va = M.tangent2ambient(x, v);
        vp = M.proj(x, va);
        v_min_vp = M.lincomb(x, 1, v, -1, vp);
        df = M.norm(x, v_min_vp);
        fprintf('Norm of tangent vector minus its projection to tangent space: %g (should be zero).\n', df);
        
        % Should check that proj is linear, self-adjoint, idempotent.
        % The issue for generic code is that manifolds do not provide means
        % to generate random vectors in the ambient space.
        
    catch up %#ok<NASGU>
        fprintf('Couldn''t check tangent2ambient, proj, norm\n');
    end    
    
    %% Checking exp and dist
    try
        x = M.rand();
        v = M.randvec(x);
        for t = logspace(-8, 1, 10)
            y = M.exp(x, v, t);
            d = M.dist(x, y);
            err = d - abs(t)*M.norm(x, v);
            fprintf(['dist(x, M.exp(x, v, t)) - abs(t)*M.norm(x, v) = ' ...
                     '%g (t = %.1e; should be zero for small enough t).\n'], ...
                     err, t);
        end
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
        fprintf(['Unless otherwise stated, M.vec seems to return real ' ...
                 'column vectors, as intended.\n']);
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
    
    %% Checking dim
    dim_threshold = 200;
    if M.dim() <= dim_threshold
        x = M.rand();
        n = M.dim() + 1;
        B = cell(n, 1);
        for k = 1 : n
            B{k} = M.randvec(x);
        end
        G = grammatrix(M, x, B);
        eigG = sort(real(eig(G)), 'descend');
        fprintf('Testing M.dim() (works best when dimension is small):\n');
        fprintf('\tIf this number is machine-precision zero, then M.dim() may be too large: %g\n', eigG(n-1));
        fprintf('\tIf this number is not machine-precision zero, then M.dim() may be too small: %g\n', eigG(n));
    else
        fprintf('M.dim() not tested because it is > %d.\n', dim_threshold);
    end
    
    %% Checking isotransp
    try
        x1 = M.rand();
        x2 = M.rand();
        u1 = M.randvec(x1);
        u2 = M.randvec(x1);
        PT_u1_x2 = M.isotransp(x1, x2, u1);
        PT_u2_x2 = M.isotransp(x1, x2, u2);

        isometry_res = M.inner(x1, u1, u2) - M.inner(x2, PT_u1_x2, PT_u2_x2);
        reverse_res = sum(sum(M.isotransp(x2, x1, PT_u1_x2) - u1));

        fprintf(['Testing isotransp(X1,X2,U) belongs to the tangent ' ...
            'space of X2: %g (should be zero).\n'], ...
            M.inner(x2, x2, PT_u1_x2));
        fprintf(['Testing isotransp is an isometry: ' ...
            '<u, v> - <isotransp(X1,X2,U), isotransp(X1,X2,V)> = %g' ...
            ' (should be zero).\n'], isometry_res);
        fprintf(['Testing isotransp reverse: ' ...
            'isotransp(X2,X1,isotransp(X1,X2,U)) - U = %g ' ...
            '(should be zero).\n'], reverse_res);

        % Parallel transport u1 to a point that is halfway along a
        % geodesic, then parallel transport to the end point. The result
        % should be the same if parallel transporting directly to the end
        % point.
        x_mid = M.exp(x1, u2, 0.5);
        x_end = M.exp(x1, u2, 1);
        PT_u1_x_mid = M.isotransp(x1, x_mid, u1);
        PT_PT_u1_x_mid_x_end = M.isotransp(x_mid, x_end, PT_u1_x_mid);
        PT_u1_x_end = M.isotransp(x1, x_end, u1);

        diff = sum(sum(PT_PT_u1_x_mid_x_end - PT_u1_x_end));
        fprintf('Testing isotransp composition: diff = %g (should be zero).\n', diff)
    catch up %#ok<NASGU>
        fprintf('Couldn''t check isotransp.\n');
    end

    %% Recommend calling checkretraction
    fprintf('It is recommended also to call checkretraction.\n');

end

