%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function Omega = makeOmegaSet_slice( n, numSlice, cross )
    
    d = length(n);

    if cross
        subs_slice = zeros( numSlice, d );
        
        for i = 1:d
            subs_slice(:,i) = randi( n(i), numSlice, 1 );
        end
        
        Omega = unique( subs_slice, 'rows' );
        m = size(Omega, 1);
        
        while m < numSlice
            subs_slice(1:m,:) = Omega;         
            for i=1:d
                 subs_slice(m+1:numSlice, i) = randi( n(i), numSlice-m, 1 );
            end
            Omega = unique( subs_slice, 'rows' );
            m = size(Omega, 1);
        end

        subs_slice = Omega;

        subs = [];
        for j = 1:numSlice
            for i = 1:d
                sub_new = repmat( subs_slice(j,:), n(i), 1 );
                subs = [subs; [sub_new(:,1:i-1), (1:n(i))', sub_new(:,i+1:end)]];
            end
        end

        Omega = subs;
        %Omega = unique( subs, 'rows' );
    else
        sizeOmega = numSlice*d;
        subs_slice = zeros( sizeOmega, d );
        
        for i = 1:d
            subs_slice(:,i) = randi( n(i), sizeOmega, 1 );
        end
        
        Omega = unique( subs_slice, 'rows' );
        m = size(Omega, 1);
        
        while m < sizeOmega
            subs_slice(1:m,:) = Omega;         
            for i=1:d
                 subs_slice(m+1:sizeOmega, i) = randi( n(i), sizeOmega-m, 1 );
            end
            Omega = unique( subs_slice, 'rows' );
            m = size(Omega, 1);
        end

        subs_slice = Omega;

        subs = [];
        for j = 1:numSlice
            for i = 1:d
                sub_new = repmat( subs_slice((j-1)*d+i,:), n(i), 1 );
                subs = [subs; [sub_new(:,1:i-1), (1:n(i))', sub_new(:,i+1:end)]];
            end
        end

        %Omega = subs;
        Omega = unique( subs, 'rows' );
        
    end

end
