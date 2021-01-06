function x = TT_to_TTeMPS( tt )
    %TT_to_TTeMPS Convert from TT Toolbox format. 
    %   A = TT_to_TTeMPS( tt ) takes the tt_tensor object tt created using the 
    %   TT Toolbox 2.x from Oseledets et al. and converts it into a TTeMPS tensor. 
    %   Toolbox needs to be installed, of course.
    %
    %   See also TTeMPS_to_TT.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    cores = {};
    ps = tt.ps;
    for i = 1:tt.d
        cores{i} = reshape( tt.core( ps(i):ps(i+1)-1 ), ...
                                [tt.r(i), tt.n(i), tt.r(i+1)] ); 
    end
    
    x = TTeMPS( cores );
    
end
