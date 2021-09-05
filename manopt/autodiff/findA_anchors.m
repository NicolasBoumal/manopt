function A = findA_anchors(problem)
% Find the indices of the anchors for the anchoredrotationsfactory
%
% function A = findA_anchors(problem)
%
% Returns the indices of the anchors for the rotation matrices manifold.
%
% See also: anchoredrotationsfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    % check availability
    assert(isfield(problem,'M'),'problem structure must contain the field M.');
    problem_name = problem.M.name();
    % The manifold must be rotation matrices with anchors
    assert(contains(problem_name,'Product rotations manifold') &&..., 
            contains(problem_name,'anchors') &&...,
            ~startsWith(problem_name,'Product manifold'),['The manifold must '... 
            'be rotation matrices with anchors'])
    % find indices of the anchors
    indexA = strfind(problem_name,'indices ') + 8;
    A = str2double(problem_name(indexA));
    for i = indexA+3:3:length(problem_name)
        A = [A,str2double(problem_name(i))];
    end

end