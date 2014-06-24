function M = productmanifold(elements)
% Returns a structure describing a product manifold M = M1 x M2 x ... x Mn.
%
% function M = productmanifold(elements)
%
% Input: an elements structure such that each field contains a manifold
% structure.
% 
% Output: a manifold structure M representing the manifold obtained by
% taking the Cartesian product of the manifolds described in the elements
% structure, with the metric obtainded by element-wise extension. Points
% and vectors are stored as structures with the same fieldnames as in
% elements.
%
% This code is for prototyping uses. The structures returned are
% inefficient representations of power manifolds but should allow to
% rapidly try out an idea.
%
% Example:
% import manopt.tools.*;
% import manopt.manifolds.sphere.*;
% M = productmanifold(struct('X', spherefactory(3), 'Y', spherefactory(4)))
% disp(M.name());
% x = M.rand()
%
% Points of M = S^2 x S^3 are represented as structures with two fields, X
% and Y. The values associated to X are points of S^2, and likewise points
% of S^3 for the field Y. Tangent vectors are also represented as
% structures with two corresponding fields X and Y.
% 
% See also: powermanifold

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    elems = fieldnames(elements);
    nelems = numel(elems);
    
    assert(nelems >= 1);
    
    M.name = @name;
    function str = name()
        str = 'Product manifold: ';
        str = [str sprintf('[%s: %s]', elems{1}, elements.(elems{1}).name())];
        for i = 2 : nelems
            str = [str sprintf(' x [%s: %s]', ...
                   elems{i}, elements.(elems{i}).name())]; %#ok<AGROW>
        end
    end
    
    M.dim = @dim;
    function d = dim()
        d = 0;
        for i = 1 : nelems
            d = d + elements.(elems{i}).dim();
        end
    end
    
    M.inner = @inner;
    function val = inner(x, u, v)
        val = 0;
        for i = 1 : nelems
            val = val + elements.(elems{i}).inner(x.(elems{i}), ...
                                               u.(elems{i}), v.(elems{i}));
        end
    end

    M.norm = @(x, d) sqrt(M.inner(x, d, d));

    M.dist = @dist;
    function d = dist(x, y)
        sqd = 0;
        for i = 1 : nelems
            sqd = sqd + elements.(elems{i}).dist(x.(elems{i}), ...
                                                           y.(elems{i}))^2;
        end
        d = sqrt(sqd);
    end
    
    M.typicaldist = @typicaldist;
    function d = typicaldist
        sqd = 0;
        for i = 1 : nelems
            sqd = sqd + elements.(elems{i}).typicaldist()^2;
        end
        d = sqrt(sqd);
    end

    M.proj = @proj;
    function v = proj(x, u)
        for i = 1 : nelems
            v.(elems{i}) = elements.(elems{i}).proj(x.(elems{i}), ...
                                                             u.(elems{i}));
        end
    end
    
    M.exp = @exp;
    function y = exp(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : nelems
            y.(elems{i}) = elements.(elems{i}).exp(x.(elems{i}), ...
                                                          u.(elems{i}), t);
        end
    end
    
    M.retr = @retr;
    function y = retr(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : nelems
            y.(elems{i}) = elements.(elems{i}).retr(x.(elems{i}), ...
                                                          u.(elems{i}), t);
        end
    end
    
    M.log = @log;
    function u = log(x1, x2)
        for i = 1 : nelems
            u.(elems{i}) = elements.(elems{i}).log(x1.(elems{i}), ...
                                                   x2.(elems{i}));
        end
    end

    % TODO: hash length will thus increase ... re-hash the thing?
    M.hash = @hash;
    function str = hash(x)
        str = '';
        for i = 1 : nelems
            str = [str elements.(elems{i}).hash(x.(elems{i}))]; %#ok<AGROW>
        end
    end

    M.lincomb = @lincomb;
    function v = lincomb(x, a1, u1, a2, u2)
        if nargin == 3
            for i = 1 : nelems
                v.(elems{i}) = elements.(elems{i}).lincomb(x.(elems{i}), ...
                                                        a1, u1.(elems{i}));
            end
        elseif nargin == 5
            for i = 1 : nelems
                v.(elems{i}) = elements.(elems{i}).lincomb(x.(elems{i}), ...
                                     a1, u1.(elems{i}), a2, u2.(elems{i}));
            end
        else
            error('Bad usage of productmanifold.lincomb');
        end
    end

    M.rand = @rand;
    function x = rand()
        for i = 1 : nelems
            x.(elems{i}) = elements.(elems{i}).rand();
        end
    end

    M.randvec = @randvec;
    function u = randvec(x)
        for i = 1 : nelems
            u.(elems{i}) = elements.(elems{i}).randvec(x.(elems{i}));
        end
        u = M.lincomb(x, 1/sqrt(nelems), u);
    end

    M.zerovec = @zerovec;
    function u = zerovec(x)
        for i = 1 : nelems
            u.(elems{i}) = elements.(elems{i}).zerovec(x.(elems{i}));
        end
    end

    M.transp = @transp;
    function v = transp(x1, x2, u)
        for i = 1 : nelems
            v.(elems{i}) = elements.(elems{i}).transp(x1.(elems{i}), ...
                                              x2.(elems{i}), u.(elems{i}));
        end
    end

    M.pairmean = @pairmean;
    function y = pairmean(x1, x2)
        for i = 1 : nelems
            y.(elems{i}) = elements.(elems{i}).pairmean(x1.(elems{i}), ...
                                                        x2.(elems{i}));
        end
    end

end
