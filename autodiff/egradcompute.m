function egrad = egradcompute(problem,x,complexflag)
% Computes the Euclidean gradient of the cost function at x using AD

% function egrad = egradcompute(autogradfunc,x,complexflag)

% Returns the Euclidean gradient of the cost function described in 
% autogradfunc at the point x.

% Note: the problem structure must contain the field autogradfunc.
% autogradfunc should be either an AcceleratedFunction or a function which 
% contains dlgradient. x is a point on the target manifold. complexflag is
% bool variable which indicates whether or not the problem described in 
% problem involves complex numbers.

% See also: autograd, mat2dl, mat2dl_complex, dl2mat, dl2mat_complex
    
    assert(isfield(problem,'autogradfunc'),['the problem structure must'...,
        ' contain the field autogradfunc, see autograd.'])
    % convert x into dlarrays to prepare for AD
    if complexflag == true
        dlx = mat2dl_complex(x);
    else
        dlx = mat2dl(x);
    end
    
    % compute egrad according to autogradfunc
    [~,egrad] = dlfeval(problem.autogradfunc,dlx);
    
    % convert egrad back to numerical arrays
    if complexflag == true
        egrad = dl2mat_complex(egrad);
    else
        egrad = dl2mat(egrad);
    end
 
end