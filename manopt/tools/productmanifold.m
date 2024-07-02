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
%
%   Jan. 4, 2021 (NB):
%       Changes for compatibility with Octave 6.1.0: by introducing a
%       "helper" function, we separate out the pre-computations. This way,
%       all pre-computed quantities are passed as input to the helper
%       function. This makes them available to nested subfunctions.
%       The extra step is not necessary in Matlab.
%
%   July  2, 2024 (NB):
%       Added check all_elements_provide() to most functions.
%       Added retr2, isotransp and paralleltransp.


    elems = fieldnames(elements);
    nelems = numel(elems);
    
    assert(nelems >= 1, ...
           'elements must be a structure with at least one field.');

    % Below are some precomputations for the mat/vec pair.
    %
    % Gather the length of the column vector representations of tangent
    % vectors for each of the manifolds. Raise a flag if any of the base
    % manifolds has no vec function available.
    vec_available = true;
    vec_lens = zeros(nelems, 1);
    for ii = 1 : nelems
        Mi = elements.(elems{ii});
        if isfield(Mi, 'vec')
            rand_x = Mi.rand();            % Assumes rand() and zerovec()
            zero_u = Mi.zerovec(rand_x);   % are available; they should be.
            vec_lens(ii) = length(Mi.vec(rand_x, zero_u));
        else
            vec_available = false;
            break;
        end
    end
    vec_pos = cumsum([1 ; vec_lens]);
    %
    vecmatareisometries = vec_available;
    for ii = 1 : nelems
        if ~isfield(elements.(elems{ii}), 'vecmatareisometries') || ...
           ~elements.(elems{ii}).vecmatareisometries()
            vecmatareisometries = false;
            break;
        end
    end
    %
    % Above are some precomputations for the mat/vec pair.
    
    % The helper function is the actual factory.
    M = productmanifoldhelper(elements, elems, nelems, vec_available, ...
                              vec_pos, vecmatareisometries);
    
end


function M = productmanifoldhelper(elements, elems, nelems, ...
                                   vec_available, vec_pos, ...
                                   vecmatareisometries)

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
       
    if all_elements_provide('name')
        M.name = @name;
    end
    function str = name()
        str = 'Product manifold: ';
        str = [str sprintf('[%s: %s]', ...
                           elems{1}, elements.(elems{1}).name())];
        for i = 2 : nelems
            str = [str sprintf(' x [%s: %s]', ...
                   elems{i}, elements.(elems{i}).name())]; %#ok<AGROW>
        end
    end
    
    if all_elements_provide('dim')
        M.dim = @dim;
    end
    function d = dim()
        d = 0;
        for i = 1 : nelems
            d = d + elements.(elems{i}).dim();
        end
    end
    
    if all_elements_provide('inner')
        M.inner = @inner;
        M.norm = @(x, d) sqrt(M.inner(x, d, d));
    end
    function val = inner(x, u, v)
        val = 0;
        for i = 1 : nelems
            val = val + elements.(elems{i}).inner(x.(elems{i}), ...
                                               u.(elems{i}), v.(elems{i}));
        end
    end

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

    if all_elements_provide('proj')
        M.proj = @proj;
    end
    function v = proj(x, u)
        for i = 1 : nelems
            v.(elems{i}) = elements.(elems{i}).proj(x.(elems{i}), ...
                                                    u.(elems{i}));
        end
    end

    if all_elements_provide('tangent')
        M.tangent = @tangent;
    end
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

    if all_elements_provide('egrad2rgrad')
        for ii = 1 : nelems
            if nargin(elements.(elems{ii}).egrad2rgrad) > 2
                warning('manopt:productmanifold:egrad2rgrad', ...
                       ['Product manifolds call M.egrad2rgrad with ', ...
                        'only two inputs:\nstoredb and key won''t be ', ...
                        'available.']);
                break;
            end
        end
        M.egrad2rgrad = @egrad2rgrad;
    end
    function g = egrad2rgrad(x, g)
        for i = 1 : nelems
            g.(elems{i}) = elements.(elems{i}).egrad2rgrad(...
                                               x.(elems{i}), g.(elems{i}));
        end
    end

    if all_elements_provide('ehess2rhess')
        for ii = 1 : nelems
            if nargin(elements.(elems{ii}).ehess2rhess) > 4
                warning('manopt:productmanifold:ehess2rhess', ...
                       ['Product manifolds call M.ehess2rhess with ', ...
                        'only four inputs:\nstoredb and key won''t be', ...
                        ' available.']);
                break;
            end
        end
        M.ehess2rhess = @ehess2rhess;
    end
    function h = ehess2rhess(x, eg, eh, h)
        for i = 1 : nelems
            h.(elems{i}) = elements.(elems{i}).ehess2rhess(...
                 x.(elems{i}), eg.(elems{i}), eh.(elems{i}), h.(elems{i}));
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
    
    if all_elements_provide('retr')
        M.retr = @retr;
    end
    function y = retr(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : nelems
            y.(elems{i}) = elements.(elems{i}).retr(x.(elems{i}), ...
                                                    u.(elems{i}), t);
        end
    end
    
    if all_elements_provide('retr2')
        M.retr2 = @retr2;
    end
    function y = retr2(x, u, t)
        if nargin < 3
            t = 1.0;
        end
        for i = 1 : nelems
            y.(elems{i}) = elements.(elems{i}).retr2(x.(elems{i}), ...
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

    if all_elements_provide('hash')
        M.hash = @hash;
    end
    function str = hash(x)
        str = '';
        for i = 1 : nelems
            str = [str elements.(elems{i}).hash(x.(elems{i}))]; %#ok<AGROW>
        end
        str = ['z' hashmd5(str)];
    end

    if all_elements_provide('lincomb')
        M.lincomb = @lincomb;
    end
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

    if all_elements_provide('rand')
        M.rand = @rand;
    end
    function x = rand()
        for i = 1 : nelems
            x.(elems{i}) = elements.(elems{i}).rand();
        end
    end

    if all_elements_provide('randvec')
        M.randvec = @randvec;
    end
    function u = randvec(x)
        for i = 1 : nelems
            u.(elems{i}) = elements.(elems{i}).randvec(x.(elems{i}));
        end
        u = M.lincomb(x, 1/sqrt(nelems), u);
    end

    if all_elements_provide('zerovec')
        M.zerovec = @zerovec;
    end
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

    if all_elements_provide('isotransp')
        M.isotransp = @isotransp;
    end
    function v = isotransp(x1, x2, u)
        for i = 1 : nelems
            v.(elems{i}) = elements.(elems{i}).isotransp(x1.(elems{i}), ...
                                              x2.(elems{i}), u.(elems{i}));
        end
    end

    if all_elements_provide('paralleltransp')
        M.paralleltransp = @paralleltransp;
    end
    function v = paralleltransp(x1, x2, u)
        for i = 1 : nelems
            v.(elems{i}) = elements.(elems{i}).paralleltransp( ...
                                        x1.(elems{i}), ...
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

    M.vecmatareisometries = @() vecmatareisometries;    
    
    if all_elements_provide('lie_identity')
        M.lie_identity = @lie_identity;
    end

    function I = lie_identity()
        I = struct();
        for i = 1 : nelems
            Mi = elements.(elems{i});
            Ii = Mi.lie_identity();
            I.(elems{i}) = Ii;
        end
    end
end
