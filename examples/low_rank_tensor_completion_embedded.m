function low_rank_tensor_completion_embedded()
% Given partial observation of a low rank tensor (possibly including noise),
% attempts to complete it.
%
% function low_rank_tensor_completion_embedded()
%
% !!! NOTE: The Tensor Toolbox Version 2.6 or higher (see https://www.tensortoolbox.org/
% or https://gitlab.com/tensors/tensor_toolbox) is needed for the execution
% of this file !!!
%
% This example demonstrates how to use the geometry factory for the
% embedded submanifold of fixed-rank tensors, embedded_tensor_factory.
%
% This geometry is described in the article
% "A Riemannian trust-region method for low-rank tensor completion"
% Gennadij Heidel and Volker Schulz, DOI:10.1002/nla.2175.
%
% This can be a starting point for many optimization problems of the form:
%
% minimize f(X) such that rank(X) = [r1 ... rd], size(X) = [n1 ... nd].
%
% Input:  None. This example file generates random data with noise.
% 
% Output: None.
%
% Please cite the Manopt and MATLAB Tensor Toolbox papers as well as the
% research paper:
%     @Article{heidel2018riemannian,
%       Title   = {A {R}iemannian trust-region method for low-rank tensor completion},
%       Author  = {G. Heidel and V. Schulz},
%       Journal = {Numerical Linear Algebra with Applications},
%       Year    = {2018},
%       Volume  = {23},
%       Number  = {6},
%       Pages   = {e1275},
%       Doi     = {10.1002/nla.2175}
%     }

% This file is part of Manopt: www.manopt.org.
% Original author: Gennadij Heidel, January 24, 2019.
% Contributors: 
% Change log:
%

    assert(exist('ttensor','file')==2,['It seems the Tensor Toolbox is ',...
    'not installed. It is needed for the execution of low_rank_tensor_completion_embedded.m. ',...
    'Please download the Toolbox first at https://www.tensortoolbox.org/',...
    ' or https://gitlab.com/tensors/tensor_toolbox.']);
    
    rng('default')
    

    % Random data generation with pseudo-random numbers from a 
    % uniform distribution on [0, 1].  
    tensor_dims = [60 40 20];
    core_dims = [8 6 5];
    total_entries = prod(tensor_dims);
    d = length(tensor_dims);
    
    % Standard deviation of normally distributed noise (set sigma to 0 to get
    % noise-free case)
    sigma = 0.1;
    
    % Generate a random tensor A of size n1-by-...-by-nd of rank (r1, ..., rd).
    U = cell(0);
    R = cell(0);
    for i=1:d
        [U{end+1},R{end+1}] = qr(randn(tensor_dims(i), core_dims(i)), 0);
    end

    Z.U = R;
    Z.G = tenrand(core_dims);
    Core = ttm(Z.G,Z.U);

    Y.U = U;
    Y.G = Core;
    A = ttm(Core,Y.U);
    
    % add noise to low-rank tensor
    A = A + sigma*tensor(randn(tensor_dims));
    
    
    % Generate a random mask P for observed entries: P(i, j, k) = 1 if the entry
    % (i, j, k) of A is observed, and 0 otherwise.
    fraction = 0.1; % Fraction of known entries.
    nr = round(fraction * total_entries);
    ind = randperm(total_entries);
    ind = ind(1 : nr);
    P = false(tensor_dims);
    P(ind) = true;
    % Hence, we know the nonzero entries in PA:
    P = tensor(P);
    PA = P.*A; 
    % Note that an efficient implementation would require evaluating A as a
    % sparse tensor only at the indices of P

    
    
    % Pick the submanifold of tensors of size n1-by-...-by-nd of rank
    % (r1, ..., rd).
    problem.M = fixedranktensorembeddedfactory(tensor_dims, core_dims);
    
    
    % Define the problem cost function. The store structure is used to minimize
    % full tensor evaluations.
    problem.cost = @cost;
    function [f,store] = cost(X, store)
        if ~isfield(store, 'PXmPA')
            Xfull = full(X.X);
            store.PXmPA = P.*Xfull - PA;
        end
        f = .5*norm(store.PXmPA)^2;
    end

    % Define the Euclidean gradient of the cost function, that is, the
    % gradient of f(X) seen as a standard function of X.
    problem.egrad =  @egrad;
    function [g,store] = egrad(X, store)
        if ~isfield(store, 'PXmPA')
            Xfull = full(X.X);
            store.PXmPA = P.*Xfull - PA;
        end
        g = store.PXmPA;
    end
    
    % Define the Euclidean Hessian of the cost at X.
    problem.ehess = @ehess;
    function [H] = ehess(X, eta)
        ambient_H = problem.M.tangent2ambient(X,eta);
        H = P.*ambient_H;
    end
    
    % Options
    X0 = problem.M.rand();
    options.maxiter = 3000;
    options.maxinner = 100;
    options.maxtime = inf;
    options.storedepth = 3;
    % Relative residual in gradient norm wanted
    store.PXmPA = P.*full(X0.X) - PA;
    options.tolgradnorm = 1e-8*problem.M.norm(X0,problem.M.egrad2rgrad(X0,problem.egrad(X0,store)));        

     % Minimize the cost function using Riemannian trust-regions
    [Xtr,~,~,~] = trustregions(problem, X0, options);

    % Postprocessing
    Xtrfull = full(Xtr.X);
    fprintf('||X-A||_F / ||A||_F = %g\n', norm(Xtrfull - A)/norm(A));
    fprintf('||PX-PA||_F / ||PA||_F = %g\n', norm(P.*Xtrfull - PA)/norm(PA));
end
