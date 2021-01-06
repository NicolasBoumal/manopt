function disp( x, name )
    %DISP Display TT/MPS tensor. 
    %   disp(X) displays the TT/MPS tensor X as a tensor network.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt


    if (nargin < 2 || ~ischar(name))
      name = inputname(1);
    end
    
    
    disp([name, ' is a TT/MPS tensor of order ', num2str(x.order), ...
                        ' with size (', num2str(x.size), ...
                        ') and ranks (', num2str(x.rank), ')']);
    disp('');

    row1 = '';
    row2 = '';
    row3 = '';

    for i=1:x.order
        row1 = [row1, sprintf('%3i--(U%2i)--', x.rank(i), i)];
        row2 = [row2, '       |    '];
        row3 = [row3, sprintf('     %3i    ', x.size(i))];   
    end
    row1 = [row1, sprintf( '%3i', x.rank(end) )]; 
    disp(row1)
    disp(row2)
    disp(row3)

end
