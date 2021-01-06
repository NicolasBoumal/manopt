%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function Omega = makeOmegaSet_mod( n, sizeOmega )
    
    if sizeOmega > prod(n)
        error('makeOmegaSet:sizeOmegaTooHigh', 'Requested size of Omega is bigger than the tensor itself!')
    end
    
    d = length(n);
    subs = zeros(sizeOmega,d);
    
    for i = 1:d
        subs(:,i) = randi( n(i), sizeOmega, 1 );
    end
    
    Omega = unique( subs, 'rows' );
    m = size(Omega, 1);
    
    while m < sizeOmega
        subs(1:m,:) = Omega;         
        for i=1:d
             subs(m+1:sizeOmega, i) = randi( n(i), sizeOmega-m, 1 );
        end
        Omega = unique( subs, 'rows' );
        m = size(Omega, 1);
    end
end
