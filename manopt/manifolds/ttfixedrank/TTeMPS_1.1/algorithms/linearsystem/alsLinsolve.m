%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function [X, residuum, cost] = alsLinsolve( L, F, X, opts )

% set default opts
if ~exist( 'opts', 'var');       opts = struct();     end
if ~isfield( opts, 'nSweeps');   opts.nSweeps = 4;  end

d = X.order;
n = X.size;


normF = norm(F);
cost = cost_function( L, X, F );
residuum = norm( apply(L, X) - F ) / normF;

for sweep = 1:opts.nSweeps
    X = orthogonalize(X, 1);
    for i = 1:d-1
        disp( ['Current core: ', num2str(i)] )

        Li = contract( L, X, i );
        Fi = contract( X, F, i );
        
        Ui = Li \ Fi(:);
        X.U{i} = reshape( Ui, size(X.U{i}) );
        X = orth_at( X, i, 'left', true );
        
        residuum = [residuum; norm( apply(L, X) - F ) / normF];
        cost = [cost; cost_function( L, X, F )];
    end
    for i = d:-1:2
        disp( ['Current core: ', num2str(i)] )

        Li = contract( L, X, i );
        Fi = contract( X, F, i );

        Ui = Li \ Fi(:);
        X.U{i} = reshape( Ui, size(X.U{i}) );
        X = orth_at( X, i, 'right', true );
        residuum = [residuum; norm( apply(L, X) - F ) / normF];
        cost = [cost; cost_function( L, X, F )];
    end

end


end

function res = cost_function( L, X, F )
res = 0.5*innerprod( X, apply(L, X) ) - innerprod( X, F );
end

function res = euclid_grad( L, X, F )
res = apply(L, X) - F;
end

