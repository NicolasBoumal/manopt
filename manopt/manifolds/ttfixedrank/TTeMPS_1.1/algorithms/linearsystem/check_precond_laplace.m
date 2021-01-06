%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function check_precond_laplace(P, grad, P_grad)

d = grad.order;


% first check whether P_grad is correctly gauged
for ii=1:d-1
    orth_err(ii) = norm( unfold(P_grad.dU{ii},'left')'*unfold(grad.U{ii},'left') , 2 ) / norm( unfold(P_grad.dU{ii},'left'), 2 );
end

if max(orth_err) > 1e-12
    orth_err
    warning('precond_laplace.m returned an inaccurate result')
end

end
