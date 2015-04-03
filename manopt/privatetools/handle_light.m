classdef handle_light < handle
% Trick to hide the methods inherited from the handle class when calling
% methods(myclass). Source:
% http://stackoverflow.com/questions/6621850/is-it-possible-to-hide-the-methods-inherited-from-the-handle-class-in-matlab
% Posted by sclarke81 on StackOverflow on Oct. 24, 2012.

   methods(Hidden)
      function lh = addlistener(varargin)
         lh = addlistener@handle(varargin{:});
      end
      function notify(varargin)
         notify@handle(varargin{:});
      end
      function delete(varargin)
         delete@handle(varargin{:});
      end
      function Hmatch = findobj(varargin)
         Hmatch = findobj@handle(varargin{:});
      end
      function p = findprop(varargin)
         p = findprop@handle(varargin{:});
      end
      function TF = eq(varargin)
         TF = eq@handle(varargin{:});
      end
      function TF = ne(varargin)
         TF = ne@handle(varargin{:});
      end
      function TF = lt(varargin)
         TF = lt@handle(varargin{:});
      end
      function TF = le(varargin)
         TF = le@handle(varargin{:});
      end
      function TF = gt(varargin)
         TF = gt@handle(varargin{:});
      end
      function TF = ge(varargin)
         TF = ge@handle(varargin{:});
      end
   end
   
end
