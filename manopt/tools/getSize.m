function bytes = getsize(variable)
% Method to get the bytes of an object.
%
% function bytes = getsize(variable)
%
% Source:
% https://ch.mathworks.com/matlabcentral/answers/14837-how-to-get-size-of-an-object
% Posted by Dmitry Borovoy on Matlab Answers on Aug. 31, 2011.
% Extended to structs by Mario Reutter on Matlab Answers on Dec. 13, 2017.

% This file is part of Manopt: www.manopt.org.
% Original authors: Dmitry Borovoy and Mario Reutter
% Contributors: Victor Liao
% Change log:
%
%   VL July 25, 2022:
%       Improve code readability and structure.

    props = properties(variable); 
    if size(props, 1) < 1
        bytes = whos(varname(variable));
        bytes = bytes.bytes;
    else % code of Dmitry
      bytes = 0;
      for ii = 1:length(props)
          currentProperty = getfield(variable, char(props(ii)));
          s = whos(varname(currentProperty));
          bytes = bytes + s.bytes;
      end
    end
end

function name = varname( ~ )
    name = inputname(1);
end
