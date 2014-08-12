function [u, lambda] = hessianescape_OBSOLETE(problem, x)
% Returns a tangent vector u that minimizes u' Hess f(x) u.
%
% TODO TODO TODO THIS IS IN THE WORKS.
% There is a potential major problem: the extra dimensions due to vec() and
% mat() not returning vectors of size problem.M.dim() create extra
% eigenvalues of value 0. But then: (1) how can we identify strictly
% positive definite Hessians? (2) When the Hessian is indeed positive
% /semi/definite, how do we know we get a proper tangent direction and not
% just crap, and also, it'll be much slower. Perhaps replace all this with
% a genuine optimization over the tangent space? Define an abstract linear
% manifold which is the tangent space at x and optimize Rayleigh there with
% manopt itself?
% 
% function [u, lambda] = hessianescape(problem, x)
%
% If the Hessian is defined in the problem structure, TODO TODO TODO.
%
% Requires the manifold description in problem.M to have the functions vec,
% mat, tangent and vecmatareisometries. See hessianspectrum.
%
% See also: hessianspectrum

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Aug. 5, 2014.
% Contributors: 
% Change log:


    if ~canGetHessian(problem)
        warning('manopt:hessianescape:nohessian', ...
                ['The Hessian appears to be unavailable.\n' ...
                 'Will try to use an approximate Hessian instead.\n'...
                 'Since this approximation may not be linear or '...
                 'symmetric,\nthe computation might fail and the '...
                 'results (if any)\nmight make no sense.']);
    end

    inner = @(u1, u2) problem.M.inner(x, u1, u2);
    vec = @(u_mat) problem.M.vec(x, u_mat);
    mat = @(u_vec) problem.M.mat(x, u_vec);
    tgt = @(u_mat) problem.M.tangent(x, u_mat);
    
    % n: size of a vectorized tangent vector
    % dim: dimension of the tangent space
    % necessarily, n >= dim.
    % The vectorized operators we build below will have at least n - dim
    % zero eigenvalues.
    n = length(vec(problem.M.zerovec(x)));
    dim = problem.M.dim();
    
    % The store structure is not updated by the getHessian call because the
    % eigs function will not take care of it. This might be worked around,
    % but for now we simply obtain the store structure built from calling
    % the cost and gradient at x and pass that one for every Hessian call.
    % This will typically be enough, seen as the Hessian is not supposed to
    % store anything new.
    storedb = struct();
    if canGetGradient(problem)
        [unused1, unused2, storedb] = getCostGrad(problem, x, struct()); %#ok<ASGLU>
    end
    
    hess = @(u_mat) tgt(getHessian(problem, x, tgt(u_mat), storedb));
    hess_vec = @(u_vec) vec(hess(mat(u_vec)));
    
    % We can only have a symmetric eigenvalue problem if the vec/mat pair
    % of the manifold is an isometry:
    vec_mat_are_isometries = problem.M.vecmatareisometries();
    if vec_mat_are_isometries
        mode = 'SA';
    else
        mode = 'SR';
    end
    

    % Do the magic here.
    eigs_opts.issym = vec_mat_are_isometries;
    eigs_opts.isreal = true;
    [u_vec, lambda] = eigs(hess_vec, n, 1, mode, eigs_opts); %#ok<NASGU>
    u = tgt(mat(u_vec));
    u = problem.M.lincomb(x, 1/problem.M.norm(x, u), u);
    lambda = inner(u, hess(u));
    
end
