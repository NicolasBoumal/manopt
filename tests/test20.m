function [volume orientation] = test20(problem)
% function [Xlsq Xhub D] = test4(n, m)
% All intputs are optional.
%
% Compute (an approximation) of the minimal volume Oriented Bounding 
% Box (OBB) problem.
%
%

% This file is part of Manopt: www.manopt.org.
% Original author: Pierre Borckmans, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    if ~exist('problem', 'var') || isempty(problem)
        problem = 'hundred';
    end
    
    clc; close all;
    
    % load the data sets
    data=load('OBB_data.mat');
    % load the specific problem data points
    datapoints = data.(problem)';
    % keep only the points on the convex hull
    % since only those are implied in the computation
    % of the oriented bounding box
    ch = convhulln(datapoints');
    ch = reshape(ch,size(ch,1)*3,1);
    datapoints = datapoints(:,unique(ch));
    
    % Create the problem structure
    problem = struct();
    problem.M = rotationsfactory(3);
    problem.cost = @(R) VolumeAxisAlignedBoundingBox(R);
    
    [Rbest, fbest, infos]=pso(problem);
    plot([infos.time],[infos.cost]);
    xlabel('CPU time'); ylabel('Best OBB volume');

    hold on
    
    % Compute the volume of the axis-aligned bounding box
    % given the data points and the orientation R
    % (simply given by the product of the span along the 3 directions)
    function [volume] = VolumeAxisAlignedBoundingBox(R)
        rotatedpoints = R*datapoints;
        volume = 1;
        for (k = 1:size(datapoints,1))
            volume = volume * ( max(rotatedpoints(k,:)) - min(rotatedpoints(k,:)));
        end
    end
end
