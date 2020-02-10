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
% Example:
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
%
%   July  4, 2013 (NB):
%       Added support for vec, mat, tangent.
%       Added support for egrad2rgrad and ehess2rhess.
%       Modified hash function to make hash strings shorter.
%
%   Dec. 17, 2018 (NB):
%       Added check all_elements_provide() to many functions, so that if,
%       for example, one of the elements does not provide exp(), then the
%       product manifold also won't provide exp(). This makes it easier for
%       tools such as, for example, checkgradient, to determine whether exp
%       is available or not.
%
%   Feb. 10, 2020 (NB):
%       Added warnings about calling egrad2rgrad and ehess2rhess without
%       storedb and key, even if some base manifolds allow them.


    elems = fieldnames(elements);
    nelems = numel(elems);
    
    assert(nelems >= 1, ...
           'elements must be a structure with at least one field.');
    
    % Handy function to check if all elements provide the necessary methods
    function answer = all_elements_provide(method_name)
        answer = false;
        for i = 1 : nelems
            if ~isfield(elements.(elems{i}), method_name)
                return;
            end
        end
        answer = true;
    end
       
    M.name = @name;
    function str = name()
        str = 'Product manifold: ';
        str = [str sprintf('[%s: %s]', ...
                           elems{1}, elements.(elems{1}).name())];
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

    if all_elements_provide('dist')
        M.dist = @dist;
    end
    function d = dist(x, y)
        sqd = 0;
        for i = 1 : nelems
            sqd = sqd + elements.(elems{i}).dist(x.(elems{i}), ...
                                                 y.(elems{i}))^2;
        end
        d = sqrt(sqd);
    end
    
    if all_elements_provide('typicaldist')
        M.typicaldist = @typicaldist;
    end
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

    M.tangent = @tangent;
    function v = tangent(x, u)
        for i = 1 : nelems
            v.(elems{i}) = elements.(elems{i}).tangent(x.(elems{i}), ...
                                                       u.(elems{i}));
        end
    end

    % True by default, false if any false encountered
    M.tangent2ambient_is_identity = true;
    for k = 1 : nelems
        if isfield(elements.(elems{k}), 'tangent2ambient_is_identity')
            if ~elements.(elems{k}).tangent2ambient_is_identity
                M.tangent2ambient_is_identity = false;
                break;
            end
        end
    end
    
    M.tangent2ambient = @tangent2ambient;
    function v = tangent2ambient(x, u)
        for i = 1 : nelems
            if isfield(elements.(elems{i}), 'tangent2ambient')
                v.(elems{i}) = ...
                    elements.(elems{i}).tangent2ambient( ...
                                               x.(elems{i}), u.(elems{i}));
            else
                v.(elems{i}) = u.(elems{i});
            end
        end
    end

    M.egrad2rgrad = @egrad2rgrad;
    function g = egrad2rgrad(x, g)
        for i = 1 : nelems
            g.(elems{i}) = elements.(elems{i}).egrad2rgrad(...
                                               x.(elems{i}), g.(elems{i}));
        end
    end
    for ii = 1 : nelems
        if nargin(elements.(elems{ii}).egrad2rgrad) > 2
            warning('manopt:productmanifold:egrad2rgrad', ...
                   ['Product manifolds call M.egrad2rgrad with only two ', ...
                    'inputs:\nstoredb and key won''t be available.']);
            break;
        end
    end

    M.ehess2rhess = @ehess2rhess;
    function h = ehess2rhess(x, eg, eh, h)
        for i = 1 : nelems
            h.(elems{i}) = elements.(elems{i}).ehess2rhess(...
                 x.(elems{i}), eg.(elems{i}), eh.(elems{i}), h.(elems{i}));
        end
    end
    for ii = 1 : nelems
        if nargin(elements.(elems{ii}).ehess2rhess) > 4
            warning('manopt:productmanifold:ehess2rhess', ...
                   ['Product manifolds call M.ehess2rhess with only two ', ...
                    'inputs:\nstoredb and key won''t be available.']);
            break;
        end
    end
    
    if all_elements_provide('exp')
        M.exp = @exp;
    end
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
    
    if all_elements_provide('log')
        M.log = @log;
    end
    function u = log(x1, x2)
        for i = 1 : nelems
            u.(elems{i}) = elements.(elems{i}).log(x1.(elems{i}), ...
                                                   x2.(elems{i}));
        end
    end

    M.hash = @hash;
    function str = hash(x)
        str = '';
        for i = 1 : nelems
            str = [str elements.(elems{i}).hash(x.(elems{i}))]; %#ok<AGROW>
        end
        str = ['z' hashmd5(str)];
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

    if all_elements_provide('transp')
        M.transp = @transp;
    end
    function v = transp(x1, x2, u)
        for i = 1 : nelems
            v.(elems{i}) = elements.(elems{i}).transp(x1.(elems{i}), ...
                                              x2.(elems{i}), u.(elems{i}));
        end
    end

    if all_elements_provide('pairmean')
        M.pairmean = @pairmean;
    end
    function y = pairmean(x1, x2)
        for i = 1 : nelems
            y.(elems{i}) = elements.(elems{i}).pairmean(x1.(elems{i}), ...
                                                        x2.(elems{i}));
        end
    end


    % Gather the length of the column vector representations of tangent
    % vectors for each of the manifolds. Raise a flag if any of the base
    % manifolds has no vec function available.
    vec_available = true;
    vec_lens = zeros(nelems, 1);
    for ii = 1 : nelems
        Mi = elements.(elems{ii});
        if isfield(Mi, 'vec')
            rand_x = Mi.rand();
            zero_u = Mi.zerovec(rand_x);
            vec_lens(ii) = length(Mi.vec(rand_x, zero_u));
        else
            vec_available = false;
            break;
        end
    end
    vec_pos = cumsum([1 ; vec_lens]);
    
    if vec_available
        M.vec = @vec;
        M.mat = @mat;
    end
    
    function u_vec = vec(x, u_mat)
        u_vec = zeros(vec_pos(end)-1, 1);
        for i = 1 : nelems
            range = vec_pos(i) : (vec_pos(i+1)-1);
            u_vec(range) = elements.(elems{i}).vec(x.(elems{i}), ...
                                                   u_mat.(elems{i}));
        end
    end

    function u_mat = mat(x, u_vec)
        u_mat = struct();
        for i = 1 : nelems
            range = vec_pos(i) : (vec_pos(i+1)-1);
            u_mat.(elems{i}) = elements.(elems{i}).mat(x.(elems{i}), ...
                                                       u_vec(range));
        end
    end

    vecmatareisometries = true;
    for ii = 1 : nelems
        if ~isfield(elements.(elems{ii}), 'vecmatareisometries') || ...
           ~elements.(elems{ii}).vecmatareisometries()
            vecmatareisometries = false;
            break;
        end
    end
    M.vecmatareisometries = @() vecmatareisometries;    

end
