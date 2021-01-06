function disp( x, name )
    %DISP Display TT/MPS block-mu tensor. 
    %   disp(X) displays the TT/MPS block-mu tensor X as a tensor network.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt


    if (nargin < 2 || ~ischar(name))
      name = inputname(1);
    end
    
    
    disp([name, ' is a TT/MPS block-', num2str(x.mu), ' (with block size ', num2str(x.p), ') tensor of order ', num2str(x.order), ...
                        ' with size (', num2str(x.size), ...
                        ') and ranks (', num2str(x.rank), ')']);
    disp('');

    row1 = '';
    row2 = '';
    row3 = '';
    row4 = '';

    for i=1:x.order
        if i == x.mu
            row1 = [row1, sprintf('      p=%i    ', x.p)];
        else
            row1 = [row1, '            '];
        end
        row2 = [row2, sprintf('%3i--(U%2i)--', x.rank(i), i)];
        row3 = [row3, '       |    '];
        row4 = [row4, sprintf('     %3i    ', x.size(i))];   
    end
    row2 = [row2, sprintf( '%3i', x.rank(end) )]; 
    disp(row1)
    disp(row2)
    disp(row3)
    disp(row4)

end
