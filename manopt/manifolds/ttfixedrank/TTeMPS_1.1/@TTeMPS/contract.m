function res = contract( x, y, idx )
    %CONTRACT Contraction of two TT/MPS tensors.
    %   Z = CONTRACT(X,Y,IDX) contracts all cores of the two TT/MPS tensors X and Y except 
    %   core IDX. Result Z is a tensor of size [X.rank(IDX),Y.ORDER(IDX),X.rank(IDX+1)].
    %
    %   RES = CONTRACT(X,Y,[IDX1, IDX2]) contracts all cores of the two TT/MPS tensors X and Y except 
    %   cores [IDX1, IDX2]. IDX1 and IDX2 must be two consecutive integers in ascending order. 
    %   Result RES is a cell array with two tensors of size [X.rank(IDX1),Y.ORDER(IDX1),Y.rank(IDX2)]
    %   and [Y.rank(IDX2),Y.ORDER(IDX2),X.rank(IDX2+1)], respectively.
    %
    %   See also INNERPROD.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    sz = size(idx);

    if min(sz) == 1
        if max(sz) == 1
            % Scalar IDX case: only one node not to contract
            if idx == 1
                right = innerprod( x, y, 'RL', idx+1 );
                res = tensorprod( y.U{idx}, right, 3 );
            elseif idx == x.order
                left = innerprod( x, y, 'LR', idx-1 );
                res = tensorprod( y.U{idx}, left, 1 );
            else
                left = innerprod( x, y, 'LR', idx-1 );
                right = innerprod( x, y, 'RL', idx+1 ); 

                res = tensorprod( y.U{idx}, left, 1 );
                res = tensorprod( res, right, 3 );
            end 

        elseif max(sz) == 2
            % Two-IDX case: two neighboring nodes to contract.
            if diff(idx) ~= 1
                error('Choose two neighboring nodes in ascending order.')
            end

            if idx(1) == 1
                % test for block format
                q = size(y.U{idx(2)}, 4);
                right = innerprod( x, y, 'RL', idx(2)+1 );
                res{1} = y.U{1};
                if q ~= 1 
                    s = size(y.U{idx(2)});
                    res{2} = reshape( permute( y.U{idx(2)}, [3 1 2 4] ), [s(3), s(1)*s(2)*q]);
                    res{2} = right*res{2};
                    res{2} = reshape( res{2}, [size(right, 1), s(1), s(2), q]);
                    res{2} = ipermute( res{2}, [3 1 2 4] );
                else
                    res{2} = tensorprod( y.U{idx(2)}, right, 3 );
                end

            elseif idx(2) == x.order
                % test for block format
                p = size(y.U{idx(1)}, 4);
                left = innerprod( x, y, 'LR', idx(1)-1 );
                if p ~= 1 
                    s = size(y.U{idx(1)});
                    res{1} = reshape( y.U{idx(1)}, [s(1), s(2)*s(3)*p]);
                    res{1} = left * res{1};
                    res{1} = reshape( res{1}, [size(left,1), s(2), s(3), p]);
                else
                    res{1} = tensorprod( y.U{idx(1)}, left, 1 );
                end
                res{2} = y.U{x.order};

            else
                left = innerprod( x, y, 'LR', idx(1)-1 );
                right = innerprod( x, y, 'RL', idx(2)+1 ); 
                % test for block format
                p = size(y.U{idx(1)}, 4);
                q = size(y.U{idx(2)}, 4);
                if p ~= 1 
                    s = size(y.U{idx(1)});
                    res{1} = reshape( y.U{idx(1)}, [s(1), s(2)*s(3)*p]);
                    res{1} = left * res{1};
                    res{1} = reshape( res{1}, [size(left,1), s(2), s(3), p]);
                    res{2} = tensorprod( y.U{idx(2)}, right, 3 );
                elseif q ~= 1
                    res{1} = tensorprod( y.U{idx(1)}, left, 1 );
                    s = size(y.U{idx(2)});
                    res{2} = reshape( permute( y.U{idx(2)}, [3 1 2 4] ), [s(3), s(1)*s(2)*q]);
                    res{2} = right*res{2};
                    res{2} = reshape( res{2}, [size(right, 1), s(1), s(2), q]);
                    res{2} = ipermute( res{2}, [3 1 2 4] );
                else
                    res{1} = tensorprod( y.U{idx(1)}, left, 1 );
                    res{2} = tensorprod( y.U{idx(2)}, right, 3 );
                end
                    
            end 
            
        else
            % Wrong IDX format.
            error('Unknown IDX format. Either scalar or two-dim. row-/column array expected.')
        end
            
    else
        error('Unknown IDX format. Either scalar or two-dim. row-/column array expected.')
    end
end
    
