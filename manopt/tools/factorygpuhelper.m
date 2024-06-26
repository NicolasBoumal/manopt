function M = factorygpuhelper(M)
% Manopt tool to help add GPU support to a manifold factory.
%
% function M = factorygpuhelper(M)
%
% Helper tool to add GPU support to factories. The input is a factory M
% created by one of Manopt's factories. The output is the same factory,
% with gather() and gpuArray() added in a number of places, following the
% logic that points, tangent vectors and ambient vectors are stored on the
% GPU (but scalars should be 'gathered' to the CPU). The name of the
% factory is also appended with '(GPU)'.
%
% This tool is typically called inside the factory itself, at the very end.
% It is not enough to call this tool: one also needs to create all arrays
% on the GPU directly. See spherefactory for an example.
%
% Let us know about issues via the forum on https://www.manopt.org. Thanks!
%
% See also: spherefactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Aug. 3, 2018.
% Contributors: 
% Change log: 
%   June 26, 2024 (NB):
%       Added checks that optional fields are indeed present in M.

    % Tag the factory name.
    M.name = @() [M.name(), ' (GPU)'];
    
    % Gathering scalar outputs: it's unclear whether this is necessary.
    M.inner = @(x, u, v) gather(M.inner(x, u, v));
    M.norm = @(x, u) gather(M.norm(x, u));

    if isfield(M, 'dist')
        M.dist = @(x, y) gather(M.dist(x, y));
    end
    
    % TODO: check that this works for manifolds whose points are not
    % matrices (but are structs or cells).
    if isfield(M, 'hash')
        M.hash = @(x) M.hash(gather(x));
    end
    
    % The vec/mat pair is mostly used in the hessianspectrum tool, where
    % the vector representation of tangent vectors is assumed to be in
    % 'normal' memory (as opposed to GPU). But it's unclear whether we
    % actually need this too.
    if isfield(M, 'vec')
        M.vec = @(x, u_mat) gather(M.vec(x, u_mat));
    end
    if isfield(M, 'mat')
        M.mat = @(x, u_vec) M.mat(x, gpuArray(u_vec));
    end

end
