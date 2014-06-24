function h = hashmd5(inp)
% Computes the MD5 hash of input data.
%
% function h = hashmd5(inp)
% 
% Returns a string containing the MD5 hash of the input variable. The input
% variable may be of any class that can be typecast to uint8 format, which
% is fairly non-restrictive.

% This file is part of Manopt: www.manopt.org.
% This code is a stripped version of more general hashing code by
% Michael Kleder, Nov 2005.
% Change log: 


    inp=inp(:);
    % convert strings and logicals into uint8 format
    if ischar(inp) || islogical(inp)
        inp=uint8(inp);
    else % convert everything else into uint8 format without loss of data
        inp=typecast(inp,'uint8');
    end

    % create hash
    x = java.security.MessageDigest.getInstance('MD5');
    x.update(inp);
    h = typecast(x.digest, 'uint8');
    h = dec2hex(h)';
    if(size(h,1))==1 % remote possibility: all hash bytes < 128, so pad:
        h = [repmat('0',[1 size(h,2)]);h];
    end
    h = lower(h(:)');
    clear x
	
end
