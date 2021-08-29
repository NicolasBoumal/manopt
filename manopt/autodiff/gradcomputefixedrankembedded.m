function grad = gradcomputefixedrankembedded(problem,x)
% Computes the Riemannian gradient of the cost function at x via AD for
% fixed-rank matices with an embedded geometry
%
% grad = gradcomputefixedrankembedded(problem,x)
%
% The first-order method follows the paper: 
% "Automatic differentiation for Riemannian optimization on low-rank matrix 
% and tensor-train manifolds", A. Novikov, M. Rakhuba, I. Oseledets, 2021
%
% Paper link: https://arxiv.org/pdf/2103.14974.pdf
%
% Please cite the Manopt paper as well as the research paper:
%    @Misc{novikov2021automatic,
%      Title         = {Automatic differentiation for Riemannian optimization on low-rank matrix and tensor-train manifolds}, 
%      Author        = {Alexander Novikov and Maxim Rakhuba and Ivan Oseledets},
%      Year          = {2021},
%      Eprint        = {2103.14974},
%      ArchivePrefix = {arXiv},
%      PrimaryClass  = {math.OC}
%    }
%
% See also: autograd

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors:
% Change log:

    % check availability
    assert(isfield(problem,'autogradfunc'),['the problem structure must'...,
        ' contain the field autogradfunc, see autograd.'])
    assert(sum(isfield(x,{'U','S','V'}))==3 &&..., 
        (contains(problem.M.name(),'rank','IgnoreCase',true)) &&...,
        (~startsWith(problem.M.name(),'Product manifold')),['The manifold'...
        'must be fixed-rank matices with an embedded geometry']);

    % convert A,B into dlarrays to prepare for AD
    A = mat2dl(x.U*x.S); B = mat2dl(x.V*x.S);
    
    % compute egrad according to autogradfunc
    [~,egrad] = dlfeval(problem.autogradfunc,x,A,B);
    
    % compute grad
    Udelta = dl2mat(egrad.A); Vdelta = dl2mat(egrad.B);
    grad.M = x.U'*Udelta;
    grad.Up = Udelta - x.U*((x.U)'*Udelta);
    grad.Vp = Vdelta - x.V*((x.V)'*Vdelta);

end