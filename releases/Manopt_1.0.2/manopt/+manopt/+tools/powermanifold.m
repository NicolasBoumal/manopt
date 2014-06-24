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
% This code is for prototyping uses. The structures returned are
% inefficient representations of power manifolds but should allow to
% rapidly try out an idea.
%
% Example (an inefficient representation of the oblique manifold (3, 10)):
% import manopt.tools.*;
% import manopt.manifolds.sphere.*;
% Mn = powermanifold(spherefactory(3), 10)
% disp(Mn.name());
% x = Mn.rand()
%
% See also: productmanifold

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    assert(n >= 1);
    
    Mn.name = @() sprintf('[%s]^%d', M.name(), n);
    
    Mn.dim = @() n*M.dim();
    
    Mn.inner = @inner;
    function val = inner(x, u, v)
        val = 0;
        for i = 1 : n
            val = val + M.inner(x{i}, u{i}, v{i});
        end
    end

    Mn.norm = @(x, d) sqrt(Mn.inner(x, d, d));

    Mn.dist = @dist;
    function d = dist(x, y)
        sqd = 0;
        for i = 1 : n
            sqd = sqd + M.dist(x{i}, y{i})^2;
        end
        d = sqrt(sqd);
    end

    Mn.typicaldist = @typicaldist;
    function d = typicaldist()
        sqd = 0;
        for i = 1 : n
            sqd = sqd + M.typicaldist()^2;
        end
        d = sqrt(sqd);
    end
    
    Mn.proj = @proj;
    function u = proj(x, u)
        for i = 1 : n
            u{i} = M.proj(x{i}, u{i});
        end
    end
    
    Mn.exp = @expo;
    function x = expo(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : n
            x{i} = M.exp(x{i}, u{i}, t);
        end
    end
    
    Mn.retr = @retr;
    function x = retr(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : n
            x{i} = M.retr(x{i}, u{i}, t);
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
    
    Mn.hash = @hash;
    function str = hash(x)
        str = '';
        for i = 1 : n
            str = [str M.hash(x{i})]; %#ok<AGROW>
        end
        str = ['z' manopt.privatetools.hashmd5(str)];
    end

    Mn.lincomb = @lincomb;
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

    Mn.rand = @rand;
    function x = rand()
        x = cell(n, 1);
        for i = 1 : n
            x{i} = M.rand();
        end
    end

    Mn.randvec = @randvec;
    function u = randvec(x)
        u = cell(n, 1);
        for i = 1 : n
            u{i} = M.randvec(x{i});
        end
        u = Mn.lincomb(x, 1/sqrt(n), u);
    end

    Mn.zerovec = @zerovec;
    function u = zerovec(x)
        u = cell(n, 1);
        for i = 1 : n
            u{i} = M.zerovec(x{i});
        end
    end

    Mn.transp = @transp;
    function u = transp(x1, x2, u)
        for i = 1 : n
            u{i} = M.transp(x1{i}, x2{i}, u{i});
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

end
