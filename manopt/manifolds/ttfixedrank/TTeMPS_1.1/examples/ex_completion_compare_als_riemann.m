% This example shows a simple comparison of two different algorithm for tensor completion:
%
%   -- ALS completion
%   -- Riemannian tensor completion (RTTC)
%
% in a very similar comparison as Figure 5.2. in
%   
%   Michael Steinlechner, Riemannian optimization for high-dimensional tensor completion,
%   Technical report, March 2015, revised December 2015. 
%   To appear in SIAM J. Sci. Comput. 
%
% See this report for more details about the algorithms and the setup. 
% The different to the therein described setup is only a reduced problem size (d, n, r) so 
% that it takes less time to compute the results.

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

rng(13);
d = 10;

ranks = [4, 6, 8];

cost = cell(1,length(ranks));
test = cell(1,length(ranks));
stats = cell(1,length(ranks));

for j = 1:length(ranks)
    r = ranks(j);
    rr = [1, r*ones(1,d-1), 1];

    nn = 20;
    n = nn*ones(1,d);

    opts = struct('maxiter', 50, 'tol', 0, 'reltol',0, 'gradtol',0);
    opts_tt = struct('maxiter', 60, 'tol', 0, 'reltol',0, 'gradtol',0);
    
    dof = d*nn*r^2;
    sizeOmega = 10*dof;
    sizeGamma = sizeOmega;
    
    Omega = makeOmegaSet_mod(n, sizeOmega);
    Gamma = makeOmegaSet_mod(n, sizeGamma);

    A = TTeMPS_rand( rr, n );
    A = 1/norm(A) * A;
    
    A_Omega = A(Omega);
    A_Gamma = A(Gamma);


    X0 = TTeMPS_rand( rr, n );
    X0 = 1/norm(X0) * X0;
    X0 = orthogonalize( X0, X0.order );

    [X,cost_als{j},test_als{j},stats_als{j}] = completion_als( A_Omega, Omega, A_Gamma, Gamma, X0, opts );
    [X,cost_tt{j},test_tt{j},stats_tt{j}] = completion_orth( A_Omega, Omega, A_Gamma, Gamma, X0, opts_tt );
end

l = lines(7);
midred = l(end,:);
darkred = brighten(l(end,:),-0.7);
lightred = brighten(midred,0.7);

midblue = l(1,:)
darkblue = brighten(midblue,-0.7);
lightblue = brighten(midblue,0.7);

subplot(1,2,1)
semilogy( test_als{1}(1:end),'color',darkred,'linewidth',2)
hold on
semilogy( test_als{2}(1:end),'color',midred,'linewidth',2)
semilogy( test_als{3}(1:end),'color',lightred,'linewidth',2)
semilogy( test_tt{1},'--','color',darkblue,'linewidth',2)
semilogy( test_tt{2},'--','color',midblue,'linewidth',2)
semilogy( test_tt{3},'--','color',lightblue,'linewidth',2)

xlabel('Iterations')
ylabel('Error on test set')
legend({'ALS, rank 4','ALS, rank 6', 'ALS, rank 8','RTTC, rank 4', 'RTTC, rank 6', 'RTTC, rank 8'})


subplot(1,2,2)
loglog( stats_als{1}.time(1:end), test_als{1}(1:end),'color',darkred,'linewidth',2)
hold on
loglog( stats_als{2}.time(1:end), test_als{2}(1:end),'color',midred,'linewidth',2)
loglog( stats_als{3}.time(1:end), test_als{3}(1:end),'color',lightred,'linewidth',2)
loglog( stats_tt{1}.time, test_tt{1},'--','color',darkblue,'linewidth',2)
loglog( stats_tt{2}.time, test_tt{2},'--','color',midblue,'linewidth',2)
loglog( stats_tt{3}.time, test_tt{3},'--','color',lightblue,'linewidth',2)
xlim([1e-1,1e3])

xlabel('Time [s]')
ylabel('Error on test set')
