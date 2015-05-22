function Q = randessential(N)
% Generates random essential matrices.
%
% function Q = randessential(N)
%
% Q is a [3x6] matrix where each [3x3] block is a uniformly distributed
% matrix.

% This file is part of Manopt: www.manopt.org.
% Original author: Roberto Tron, Aug. 8, 2014
% Contributors:
% Change log:
    
    if nargin < 1
        N = 1;
    end
    
    Q = [randrot(3,N) randrot(3,N)];
    
end