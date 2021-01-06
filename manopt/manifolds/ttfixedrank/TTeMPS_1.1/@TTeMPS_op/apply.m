function y = apply( A, x, idx )
    %APPLY Application of TT/MPS operator to a TT/MPS tensor
    %   Y = APPLY(A, X) applies the TT/MPS operator A to the TT/MPS tensor X.
    %
    %   Y = APPLY(A, X, idx) is the application of A but only in mode idx.
    %       note that in this case, X is assumed to be a standard matlab array and
    %       not a TTeMPS tensor. 
    %
    %   In both cases, X can come from a block-TT format, that is, with a four-dimensional core instead.
    %
    %   See also CONTRACT

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    % first case: all cores.
    if ~exist( 'idx', 'var' )
        V = cell(1, A.order);

        if isa( x, 'TTeMPS' )
            for i = 1:A.order
                %check for possible block format
                p = size(x.U{i},4);
                if p ~= 1
                    Xi = permute( x.U{i}, [2 1 3 4]);
                    Xi = reshape( Xi, [A.size_row(i), x.rank(i)*x.rank(i+1)*p] );
                    Ai = reshape( permute( A.U{i}, [1 2 4 3]), A.rank(i)*A.size_col(i)*A.rank(i+1), A.size_row(i));
                    V{i} = Ai*Xi;
                    V{i} = reshape( V{i}, [A.rank(i), A.rank(i+1), A.size_col(i), x.rank(i), x.rank(i+1), p] );
                    V{i} = permute( V{i}, [1, 4, 3, 2, 5, 6]);
                    V{i} = reshape( V{i}, [A.rank(i)*x.rank(i), A.size_col(i), A.rank(i+1)*x.rank(i+1), p]);
                else
                    Xi = matricize( x.U{i}, 2);
                    Ai = reshape( permute( A.U{i}, [1 2 4 3]), A.rank(i)*A.size_col(i)*A.rank(i+1), A.size_row(i));
                    V{i} = Ai*Xi;
                    V{i} = reshape( V{i}, [A.rank(i), A.size_col(i), A.rank(i+1), x.rank(i), x.rank(i+1)] );
                    V{i} = permute( V{i}, [1, 4, 2, 3, 5]);
                    V{i} = reshape( V{i}, [A.rank(i)*x.rank(i), A.size_col(i), A.rank(i+1)*x.rank(i+1)]);
                end
            end
            y = TTeMPS( V );

        elseif isa( x, 'TTeMPS_block' )
            mu = x.mu;
            p = x.p;
            % first case: all cores.
            V = cell(1, A.order);

            for i = [1:mu-1, mu+1:A.order]
                Xi = matricize( x.U{i}, 2);
                Ai = reshape( permute( A.U{i}, [1 2 4 3]), A.rank(i)*A.size_col(i)*A.rank(i+1), A.size_row(i));
                V{i} = Ai*Xi;
                V{i} = reshape( V{i}, [A.rank(i), A.size_col(i), A.rank(i+1), x.rank(i), x.rank(i+1)] );
                V{i} = permute( V{i}, [1, 4, 2, 3, 5]);
                V{i} = reshape( V{i}, [A.rank(i)*x.rank(i), A.size_col(i), A.rank(i+1)*x.rank(i+1)]);
            end

            Xi = permute( x.U{mu}, [2 1 3 4]);
            Xi = reshape( Xi, [A.size_row(mu), x.rank(mu)*x.rank(mu+1)*p] );
            Ai = reshape( permute( A.U{mu}, [1 2 4 3]), A.rank(mu)*A.size_col(mu)*A.rank(mu+1), A.size_row(mu));
            V{mu} = Ai*Xi;
            V{mu} = reshape( V{mu}, [A.rank(mu), A.rank(mu+1), A.size_col(mu), x.rank(mu), x.rank(mu+1), p] );
            V{mu} = permute( V{mu}, [1, 4, 3, 2, 5, 6]);
            V{mu} = reshape( V{mu}, [A.rank(mu)*x.rank(mu), A.size_col(mu), A.rank(mu+1)*x.rank(mu+1), p]);

            y = TTeMPS_block( V, mu, p );

        else
            error('Unsupported class type of vector argument. Must be TTeMPS or TTeMPS_block object')
        end

    else
        %check for possible block format
        p = size(x,4);
        if p ~= 1
            Xi = permute( x, [2 1 3 4]);
            Xi = reshape( Xi, [A.size_row(idx), size(x, 1)*size(x, 3)*p] );
            Ai = reshape( permute( A.U{idx}, [1 2 4 3]), A.rank(idx)*A.size_col(idx)*A.rank(idx+1), A.size_row(idx));
            V = Ai*Xi;
            V = reshape( V, [A.rank(idx), A.rank(idx+1), A.size_col(idx), size(x, 1), size(x, 3), p] );
            V = permute( V, [1, 4, 3, 2, 5, 6]);
            y = reshape( V, [A.rank(idx)*size(x, 1), A.size_col(idx), A.rank(idx+1)*size(x, 3), p]);
        else
            Xi = matricize( x, 2);
            Ai = reshape( permute( A.U{idx}, [1 2 4 3]), A.rank(idx)*A.size_col(idx)*A.rank(idx+1), A.size_row(idx));
            V = Ai*Xi;
            V = reshape( V, [A.rank(idx), A.size_col(idx), A.rank(idx+1), size(x, 1), size(x, 3)] );
            V = permute( V, [1, 4, 2, 3, 5]);
            y = reshape( V, [A.rank(idx)*size(x,1), A.size_col(idx), A.rank(idx+1)*size(x,3)]);
        end
    end
end
