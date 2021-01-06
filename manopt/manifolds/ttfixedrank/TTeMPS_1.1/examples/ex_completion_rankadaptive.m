% Example script for RANK-ADAPTIVE TENSOR COMPLETION, see Algorithm RTTC described in  
%
%   Michael Steinlechner, Riemannian optimization for high-dimensional tensor completion,
%   Technical report, March 2015, revised December 2015. 
%   To appear in SIAM J. Sci. Comput. 
%
% See this report for more details about the algorithms and the setup. 
% In particular, the choices of 
%   
%   maxiter, maxiter_final,
%   tol, reltol, gradtol
%
% can significantly influence the performance. They have to be chosen in such a
% way so that the algorithm does not stay too long at each intermediate rank
% (usually, less than 10 iteration per intermediate rank suffice completely).
% The correct choice requires some testing w.r.t. to the underlying data.

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

rng(17);

d = 5;
nn = 20;
maxrank = 7;
L = 1;

n = nn*ones(1,d);

opts_cg = struct('maxiter', 10,'maxiter_final',10, 'tol', 1e-6, 'reltol', 1e-6, 'gradtol', 0, 'maxrank', maxrank,'epsilon',1e-8);

dof = d*nn*maxrank^2;
sizeOmega = 10*dof;
sizeGamma = sizeOmega;

Omega = makeOmegaSet_mod(n, sizeOmega);
sizeOmega_C = 100;
sizeOmega = sizeOmega - sizeOmega_C;
Omega_C_ind = randperm( sizeOmega, sizeOmega_C );
Omega_C = Omega( Omega_C_ind, : );
Omega( Omega_C_ind, : ) = [];
Gamma = makeOmegaSet_mod(n, sizeGamma);

A_Omega = zeros(sizeOmega,1);
A_Omega_C = zeros(sizeOmega_C,1);
A_Gamma = zeros(sizeGamma,1);

f = @(x) exp(-norm(x));

for i = 1:sizeOmega
    A_Omega(i) = f( Omega(i,:)/(max(n)-1)*L );
end
for i = 1:sizeOmega_C
    A_Omega_C(i) = f( Omega_C(i,:)/(max(n)-1)*L );
end
for i = 1:sizeGamma
    A_Gamma(i) = f( Gamma(i,:)/(max(n)-1)*L );
end

r = [1, 1*ones(1,d-1), 1];
X0 = TTeMPS_rand( r, n );
X0 = orthogonalize( X0, X0.order );

[X,cost,test,stats] = completion_rankincrease( 'GeomCG', A_Omega, Omega, A_Omega_C, Omega_C, A_Gamma, Gamma, X0, opts_cg );

stats.rankidx = cumsum(stats.rankidx)
subplot(1,2,1)
semilogy( 1:length(cost), cost,'Markersize',8);
hold on
line = [1e-6,1e0];
for i=1:length(stats.rankidx)
    semilogy( [stats.rankidx(i), stats.rankidx(i)], line, '--','color',[0.7,0.7,0.7]);
end
title('Reduction of cost function')
xlabel('Number of individual RTTC iterations performed')
ylabel('Cost function')
legend('Cost function','rank increase in one mode')
set(gca,'fontsize',16)

subplot(1,2,2)
semilogy( 1:length(test), test,'Markersize',8);
title('Reduction of rel. error on test set')
xlabel('Number of full RTTC runs')
ylabel('Rel. error after one RTTC run for a certain rank')
set(gca,'fontsize',16)
