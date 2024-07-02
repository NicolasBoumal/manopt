function Mn = powermanifold(M, n)
% Returns a structure describing a power manifold M^n = M x M x ... x M.
%
% function Mn = powermanifold(M, n)
%
% Input: a manifold structure M and an integer n >= 1.
% 
% Output: a manifold structure Mn representing M x ... x M (n copies of M)
% with the metric of M extended element-wise. Points and vectors are stored
% as cells of size nx1.
%
% This code is for prototyping uses. The structures returned are often
% inefficient representations of power manifolds owing to their use of
% for-loops, but they should allow to rapidly try out an idea.
%
% Example (an inefficient representation of the oblique manifold (3, 10)):
% Mn = powermanifold(spherefactory(3), 10)
% disp(Mn.name());
% x = Mn.rand()
%
% See also: productmanifold

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log:
%
%   July  4, 2013 (NB):
%       Added support for vec, mat, tangent.
%       Added support for egrad2rgrad and ehess2rhess.
%
%   Feb. 10, 2020 (NB):
%       Added warnings about calling egrad2rgrad and ehess2rhess without
%       storedb and key, even if the base manifold allows them.
%
%   Jan.  4, 2021 (NB):
%       Changes for compatibility with Octave 6.1.0: see len_vec.
%
%   Mar.  7, 2023 (CC):
%       Include exp, dist, typicaldist only if defined in base manifold.
%
%   July  2, 2024 (NB):
%       Extended the work of the previous change to all functions.
%       Added retr2, isotransp and paralleltransp.

    
    assert(n >= 1, 'n must be an integer larger than or equal to 1.');
    
    if isfield(M, 'name')
        Mn.name = @() sprintf('[%s]^%d', M.name(), n);
    end
    
    if isfield(M, 'dim')
        dim = n*M.dim();
        Mn.dim = @() dim;
    end
    
    if isfield(M, 'inner')
        Mn.inner = @inner;
        Mn.norm = @(x, d) sqrt(inner(x, d, d));
    end
    function val = inner(x, u, v)
        val = 0;
        for i = 1 : n
            val = val + M.inner(x{i}, u{i}, v{i});
        end
    end


    if isfield(M, 'dist')
        Mn.dist = @dist;
    end
    function d = dist(x, y)
        sqd = 0;
        for i = 1 : n
            sqd = sqd + M.dist(x{i}, y{i})^2;
        end
        d = sqrt(sqd);
    end

    if isfield(M, 'typicaldist')
        Mn.typicaldist = @typicaldist;
    end
    function d = typicaldist()
        sqd = 0;
        for i = 1 : n
            sqd = sqd + M.typicaldist()^2;
        end
        d = sqrt(sqd);
    end
    
    if isfield(M, 'proj')
        Mn.proj = @proj;
    end
    function u = proj(x, u)
        for i = 1 : n
            u{i} = M.proj(x{i}, u{i});
        end
    end
    
    if isfield(M, 'tangent')
        Mn.tangent = @tangent;
    end
    function u = tangent(x, u)
        for i = 1 : n
            u{i} = M.tangent(x{i}, u{i});
        end
    end
    
    if isfield(M, 'tangent2ambient_is_identity')
        Mn.tangent2ambient_is_identity = M.tangent2ambient_is_identity;
    else
        Mn.tangent2ambient_is_identity = true;
    end
    
    if isfield(M, 'tangent2ambient')
        Mn.tangent2ambient = @tangent2ambient;
    else
        Mn.tangent2ambient = @(x, u) u;
    end
    function u = tangent2ambient(x, u)
        for i = 1 : n
            u{i} = M.tangent2ambient(x{i}, u{i});
        end
    end
    
    if isfield(M, 'egrad2rgrad')
        if nargin(M.egrad2rgrad) > 2
            warning('manopt:powermanifold:egrad2rgrad', ...
                   ['Power manifolds call M.egrad2rgrad with only two', ...
                    ' inputs:\nstoredb and key won''t be available.']);
        end
        Mn.egrad2rgrad = @egrad2rgrad;
    end
    function g = egrad2rgrad(x, g)
        for i = 1 : n
            g{i} = M.egrad2rgrad(x{i}, g{i});
        end
    end
    
    if isfield(M, 'ehess2rhess')
        if nargin(M.ehess2rhess) > 4
            warning('manopt:powermanifold:ehess2rhess', ...
                   ['Power manifolds call M.ehess2rhess with only ', ...
                    'four inputs:\nstoredb and key won''t be available.']);
        end
        Mn.ehess2rhess = @ehess2rhess;
    end
    function h = ehess2rhess(x, eg, eh, h)
        for i = 1 : n
            h{i} = M.ehess2rhess(x{i}, eg{i}, eh{i}, h{i});
        end
    end
    
    if isfield(M, 'exp')
        Mn.exp = @expo;
    end
    function x = expo(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : n
            x{i} = M.exp(x{i}, u{i}, t);
        end
    end
    
    if isfield(M, 'retr')
        Mn.retr = @retr;
    end
    function x = retr(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : n
            x{i} = M.retr(x{i}, u{i}, t);
        end
    end
    
    if isfield(M, 'retr2')
        Mn.retr2 = @retr2;
    end
    function x = retr2(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : n
            x{i} = M.retr2(x{i}, u{i}, t);
        end
    end
    
    if isfield(M, 'log')
        Mn.log = @loga;
    end
    function u = loga(x, y)
        u = cell(n, 1);
        for i = 1 : n
            u{i} = M.log(x{i}, y{i});
        end
    end
    
    if isfield(M, 'hash')
        Mn.hash = @hash;
    end
    function str = hash(x)
        str = '';
        for i = 1 : n
            str = [str, M.hash(x{i})]; %#ok<AGROW>
        end
        str = ['z' hashmd5(str)];
    end

    if isfield(M, 'lincomb')
        Mn.lincomb = @lincomb;
    end
    function x = lincomb(x, a1, u1, a2, u2)
        if nargin == 3
            for i = 1 : n
                x{i} = M.lincomb(x{i}, a1, u1{i});
            end
        elseif nargin == 5
            for i = 1 : n
                x{i} = M.lincomb(x{i}, a1, u1{i}, a2, u2{i});
            end
        else
            error('Bad usage of powermanifold.lincomb');
        end
    end

    if isfield(M, 'rand')
        Mn.rand = @rand;
    end
    function x = rand()
        x = cell(n, 1);
        for i = 1 : n
            x{i} = M.rand();
        end
    end

    if isfield(M, 'randvec')
        Mn.randvec = @randvec;
    end
    function u = randvec(x)
        u = cell(n, 1);
        for i = 1 : n
            u{i} = M.randvec(x{i});
        end
        u = lincomb(x, 1/sqrt(n), u);
    end

    if isfield(M, 'zerovec')
        Mn.zerovec = @zerovec;
    end
    function u = zerovec(x)
        u = cell(n, 1);
        for i = 1 : n
            u{i} = M.zerovec(x{i});
        end
    end

    if isfield(M, 'transp')
        Mn.transp = @transp;
    end
    function u = transp(x1, x2, u)
        for i = 1 : n
            u{i} = M.transp(x1{i}, x2{i}, u{i});
        end
    end

    if isfield(M, 'isotransp')
        Mn.isotransp = @isotransp;
    end
    function u = isotransp(x1, x2, u)
        for i = 1 : n
            u{i} = M.isotransp(x1{i}, x2{i}, u{i});
        end
    end

    if isfield(M, 'paralleltransp')
        Mn.paralleltransp = @paralleltransp;
    end
    function u = paralleltransp(x1, x2, u)
        for i = 1 : n
            u{i} = M.paralleltransp(x1{i}, x2{i}, u{i});
        end
    end

    if isfield(M, 'pairmean')
        Mn.pairmean = @pairmean;
    end
    function y = pairmean(x1, x2)
        y = cell(n, 1);
        for i = 1 : n
            y{i} = M.pairmean(x1{i}, x2{i});
        end
    end

    % Compute the length of a vectorized tangent vector of M at x, assuming
    % this length is independent of the point x (that should be fine).
    if isfield(M, 'vec')
        rand_x = M.rand();            % This assumes rand() and zerovec()
        zero_u = M.zerovec(rand_x);   % are available; they really should.
        len_vec = length(M.vec(rand_x, zero_u));
        
        Mn.vec = @(x, u_mat) vec(x, u_mat, len_vec, M, n);
        
        if isfield(M, 'mat')
            Mn.mat = @(x, u_vec) mat(x, u_vec, len_vec, M, n);
        end
        
    end
    
    function u_vec = vec(x, u_mat, len_vec, M, n)
        u_vec = zeros(len_vec, n);
        for i = 1 : n
            u_vec(:, i) = M.vec(x{i}, u_mat{i});
        end
        u_vec = u_vec(:);
    end

    function u_mat = mat(x, u_vec, len_vec, M, n)
        u_mat = cell(n, 1);
        u_vec = reshape(u_vec, len_vec, n);
        for i = 1 : n
            u_mat{i} = M.mat(x{i}, u_vec(:, i));
        end
    end

    if isfield(M, 'vecmatareisometries')
        Mn.vecmatareisometries = M.vecmatareisometries;
    else
        Mn.vecmatareisometries = @() false;
    end

    if isfield(M, 'lie_identity')
        Mn.lie_identity = @lie_identity;
    end
    function I = lie_identity()
        I_M = M.lie_identity();
        I = cell(n, 1);
        for k = 1 : n
            I{k} = I_M;
        end
    end

end
