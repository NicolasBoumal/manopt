function disp( x, name )
    %DISP Display TT/MPS operator. 
    %   disp(X) displays the TT/MPS operator X as a tensor network.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    if (nargin < 2 || ~ischar(name))
      name = inputname(1);
    end
    
    
    disp([name, ' is a TT/MPS operator of order ', num2str(x.order), ...
                ' with column sizes (', num2str(x.size_col), '),']);
    disp(['row sizes (', num2str(x.size_row), ...
                        '), and ranks (', num2str(x.rank), ')']);
    disp(' ');
    disp('');

    row0 = '';
    row1 = '';
    row2 = '';
    row3 = '';
    row4 = '';

    for i=1:x.order
        row0 = [row0, sprintf('     %3i    ', x.size_col(i))]; 
        row1 = [row1, '       |    '];
        row2 = [row2, sprintf('%3i--(U%2i)--', x.rank(i), i)];
        row3 = [row3, '       |    '];
        row4 = [row4, sprintf('     %3i    ', x.size_row(i))];  
    end
    row2 = [row2, sprintf( '%3i', x.rank(end) )]; 
    disp(row0)
    disp(row1)
    disp(row2)
    disp(row3)
    disp(row4)

end
