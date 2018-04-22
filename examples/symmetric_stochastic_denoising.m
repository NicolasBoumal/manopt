function symmetric_stochastic_denoising()
% Given a symmetric matrix, find its closest, in the Frobenius norm,  
% symmetric stochastic matrix.
%
% function symmetric_stochastic_denoising()
%
% This example demonstrates how to use the geometry factory for the
% symmetric stochastic multinomial manifold, symmultinomialfactory.
% The file is based on developments in the research paper
% A. Douik and B. Hassibi, "Manifold Optimization Over the Set 
% of Doubly Stochastic Matrices: A Second-Order Geometry"
% ArXiv:1802.02628, 2018.
%
% Link to the paper: https://arxiv.org/abs/1802.02628.
%
% Please cite the Manopt paper as well as the research paper:
%     @Techreport{Douik2018Manifold,
%       Title   = {Manifold Optimization Over the Set of Doubly Stochastic 
%           Matrices: {A} Second-Order Geometry},
%       Author  = {Douik, A. and Hassibi, B.},
%       Journal = {Arxiv preprint ArXiv:1802.02628},
%       Year    = {2018}
%     }
% This can be a starting point for many optimization problems of the form:
%
% minimize f(X) such that X is a symmetric stochastic matrix.
%
% Note that the code is long because it showcases quite a few features of
% Manopt: most of the code is optional.
%
% Input:  None. This example file generates random data.
% 
% Output: None.
% This file is part of Manopt: www.manopt.org.
% Original author: Ahmed Douik, March 15, 2018.
% Contributors:
% Change log:

    close all
    clear all
    clc 
    warning('off', 'manopt:multinomialsymmetricfactory:exp') 
    % Clearing all variables, open windows, and disabiling all warnings 
    % regarding the use of a retraction instead of the exponential map.
    
    symm = @(X) .5*(X+X'); % Inline function to make a matrix symmetric

    % Input data
    n = 100 ; % Size of the matrix
    sigma = 1/n^2 ; % Noise standard deviation at each entry

    % Generate a symmetric stochastic matrix using the DAD algorithm
    B = symm(doubly_stochastic(abs(randn(n, n)))) ;
    % Adding noise to the matrix
    A = max(B + sigma*symm(randn(n, n)),0.01) ;

    % Denoising function and derivatives
    cost = @(X) (0.5*norm( A-X, 'fro')^2)  ; 
    egrad = @(X) (X -A )  ; 
    ehess = @(X, U) (U) ;

    % Manifold initialization
    manifold = multinomialsymmetricfactory(n) ;
    problem.M = manifold;
    problem.cost = cost;
    problem.grad = @(X) manifold.egrad2rgrad(X, egrad(X));
    problem.hess = @(X, U) manifold.ehess2rhess(X, egrad(X), ehess(X, U), U);

    % Check consistency of the gradient and the Hessian. Useful if you
    % adapt this example for a new cost function and you would like to make
    % sure there is no mistake.
    figure() ;
    checkgradient(problem); % Check the gradient
    figure() ;
    checkhessian(problem);  % Check the hessian. This test usually fails
                            % due to the use of a first order retraction, 
                            % unless the test is performed at the optimal.
    
    % A first order algorithm
    % Minimize the cost function using the Conjugate Gradient algorithm. 
    [X1, ~, info, ~] = conjugategradient(problem); 
    S1 = [info.gradnorm] ; % Collecting the Gradient information
    
    % Computing the distance between the original, noisyless matrix and the
    % found solution
    fprintf('||X-B||_F^2 = %g\n', norm(X1 - B, 'fro')^2);

    % A second order algorithm
    % Minimize the cost function using the trust-regions method. 
    [X2, ~, info, ~] = trustregions(problem);                                     
    S2 = [info.gradnorm] ; % Collecting the Gradient information

    % Computing the distance between the original, noisyless matrix and the
    % found solution
    fprintf('||X-B||_F^2 = %g\n', norm(X2 - B, 'fro')^2);
    
    figure() ;
    loglog(S1,'red' ,'Marker','*','LineWidth',2,'DisplayName','Conjugate Gradient Algorithm') ; 
    hold on ;
    loglog(S2,'blue' ,'Marker','+','LineWidth',2,'DisplayName','Trust Regions Method') ; 
    
    set(gca,'YGrid','on','XGrid','on') ;
    xlabel('Number of Iteration','FontSize',16);
    ylabel('Norm of Gradient','FontSize',16);
    legend1 = legend('show');
    set(legend1,'FontSize',16);
    
    % If the problem has a small enough dimension, we may (for analysis
    % purposes) compute the spectrum of the Hessian at a point X. This may
    % help in studying the conditioning of a problem. If you don't provide
    % the Hessian, Manopt will approximate the Hessian with finite
    % differences of the gradient and try to estimate its "spectrum" (it's
    % not a proper linear operator). This can give some intuition, but
    % should not be relied upon.
    if problem.M.dim() < 100
        fprintf('Computing the spectrum of the Hessian...');
        s1 = hessianspectrum(problem, X1);
        figure() ;
        hist(s1);
    end
    
    figure()
    checkhessian(problem,X1) % Checking the hessian at the optimal point. 
                             % Notice that is the optimal solution has 
                             % zeros, then it is not achievable by the 
                             % manifold and the test is expected to fail.
end


