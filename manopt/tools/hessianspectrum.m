function lambdas = hessianspectrum(problem, x, storedb, key)
% Returns the eigenvalues of the (preconditioned) Hessian at x.
% 
% function lambdas = hessianspectrum(problem, x)
% function lambdas = hessianspectrum(problem, x, storedb)
% function lambdas = hessianspectrum(problem, x, storedb, key)
%
% If the Hessian is defined in the problem structure and if no
% preconditioner is defined, returns the eigenvalues of the Hessian
% operator (which needs to be symmetric but not necessarily definite) on
% the tangent space at x. There are problem.M.dim() eigenvalues.
%
% If a preconditioner is defined in the problem structure, the eigenvalues
% of the composition is computed: Precon o Hessian. Remember that the
% preconditioner has to be symmetric, positive definite, and is supposed to
% approximate the inverse of the (Riemannian) Hessian.
%
% The typical ways to define a preconditioner are via problem.precon or
% problem.sqrtprecon (see comment below). These should be function handles
% with the same input/output system as problem.hess for the Hessian.
%
% Even though the Hessian and the preconditioner are both symmetric, their
% composition is not symmetric, which can slow down the call to 'eigs'
% substantially. If possible, you may specify the square root of the
% preconditioner in the problem structure, as sqrtprecon. This operator on
% the tangent space at x must also be symmetric, positive definite, and
% such that sqrtprecon o sqrtprecon = precon. Then, the spectrum of the
% symmetric operator sqrtprecon o hess o sqrtprecon is computed: it is the
% same as the spectrum of precon o hess, but is usually faster to compute.
%
% The input and the output of the Hessian and of the preconditioner are
% projected on the tangent space to avoid undesired contributions of the
% ambient space.
%
% storedb is a StoreDB object, key is the StoreDB key to point x.
%
% Requires the manifold description in problem.M to have these functions:
% 
%   u_vec = vec(x, u_mat) :
%       Returns a column vector representation of the normal (usually
%       matrix) representation of the tangent vector u_mat. vec must be an
%       isometry between the tangent space (with its Riemannian metric) and
%       a subspace of R^n where n = length(u_vec), with the 2-norm on R^n.
%       In other words: it is an orthogonal projector.
%
%   u_mat = mat(x, u_vec) :
%       The inverse of vec (its adjoint).
%
%   u_mat_clean = tangent(x, u_mat) :
%       Subtracts from the tangent vector u_mat any component that would
%       make it "not really tangent", by projection.
%
%   answer = vecmatareisometries() :
%       Returns true if the linear maps encoded by vec and mat are
%       isometries, false otherwise. It is better if the answer is yes.
%
% See also: hessianextreme canGetPrecon canGetSqrtPrecon

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 3, 2013.
% Contributors: 
% Change log:
%
%   Dec. 18, 2014 (NB):
%       The lambdas are now sorted when they are returned.
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%       Does no longer accept sqrtprecon as an input: the square root of
%       the preconditioner may now be specified directly in the problem
%       structure, following the same syntax as the preconditioner precon.

    % Allow omission of the key, and even of storedb.
    if ~exist('storedb', 'var')
        storedb = StoreDB();
    end
    if ~exist('key', 'var')
        key = storedb.getNewKey();
    end


    if ~canGetHessian(problem)
        warning('manopt:hessianspectrum:nohessian', ...
                ['The Hessian appears to be unavailable.\n' ...
                 'Will try to use an approximate Hessian instead.\n'...
                 'Since this approximation may not be linear or '...
                 'symmetric,\nthe computation might fail and the '...
                 'results (if any)\nmight make no sense.']);
    end

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
    
    % It is usually a good idea to force a gradient computation to make
    % sure precomputable things are precomputed.
    if canGetGradient(problem)
        [unused1, unused2] = getCostGrad(problem, x, storedb, key); %#ok
    end
    
    hess = @(u_mat) tgt(getHessian(problem, x, tgt(u_mat), storedb, key));
    hess_vec = @(u_vec) vec(hess(mat(u_vec)));
    
    % Regardless of preconditioning, we can only have a symmetric
    % eigenvalue problem if the vec/mat pair of the manifold is an
    % isometry:
    vec_mat_are_isometries = problem.M.vecmatareisometries();
    
    if ~canGetSqrtPrecon(problem)
    
        if ~canGetPrecon(problem)
            
            % There is no preconditioner: just deal with
			% the (symmetric) Hessian.
            
            eigs_opts.issym = vec_mat_are_isometries;
            eigs_opts.isreal = true;
            lambdas = eigs(hess_vec, n, dim, 'LM', eigs_opts);
            
        else
            
            % There is a preconditioner, but we don't have its square root:
            % deal with the non-symmetric composition prec o hess.
            
            prec = @(u_mat) tgt(getPrecon(problem, x, tgt(u_mat), storedb, key));
            prec_vec = @(u_vec) vec(prec(mat(u_vec)));
            % prec_inv_vec = @(u_vec) pcg(prec_vec, u_vec);

            eigs_opts.issym = false;
            eigs_opts.isreal = true;
            lambdas = eigs(@(u_vec) prec_vec(hess_vec(u_vec)), ...
                           n, dim, 'LM', eigs_opts);
            
        end
        
    else
        
        % There is a preconditioner, and we have its square root: deal with
        % the symmetric composition sqrtprecon o hess o sqrtprecon.
        
        sqrtprec = @(u_mat) tgt(getSqrtPrecon(problem, x, tgt(u_mat), storedb, key));
        sqrtprec_vec = @(u_vec) vec(sqrtprec(mat(u_vec)));
        
        eigs_opts.issym = vec_mat_are_isometries;
        eigs_opts.isreal = true;
        lambdas = eigs(@(u_vec) ...
                      sqrtprec_vec(hess_vec(sqrtprec_vec(u_vec))), ...
                      n, dim, 'LM', eigs_opts);
        
    end
    
    lambdas = sort(lambdas);

end
