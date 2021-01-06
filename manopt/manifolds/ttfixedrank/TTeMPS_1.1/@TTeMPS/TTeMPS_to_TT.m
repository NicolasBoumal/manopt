function TT = TTeMPS_to_TT( x )
    %TTeMPS_to_TT Convert to TT Toolbox format. 
    %   TT = TT_to_TTeMPS( A ) takes the TTeMPS tensor A and converts it into
    %   a tt_tensor object tt using the TT Toolbox 2.x from Oseledets et al.
    %   Toolbox needs to be installed, of course.
    %
    %   See also TTeMPS_to_TT.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    TT = tt_tensor;
    TT.d = x.order;
    TT.n = x.size';
    TT.r = x.rank';
    
    % cellfun goodness!
    TT.core = cell2mat( cellfun(@(y) y(:).', x.U, 'UniformOutput', false) ).';
    lengths = TT.r(1:end-1) .* TT.n .* TT.r(2:end);
    TT.ps = cumsum( [1; lengths] );
    
end
