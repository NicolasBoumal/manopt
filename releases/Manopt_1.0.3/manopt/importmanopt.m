% Add Manopt to the path and import all manopt components.
% Among other things, this is helpful to call once in the command line, so
% that help will be able to locate manopt files.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Jan. 3, 2013.
% Contributors: 
% Change log: 

% Determine the path to the Manopt directory, which should be the directory
% that this script is in.
thisfilefull = mfilename('fullpath');
thisfilename = mfilename();
thisfilepath = thisfilefull(1:strfind(thisfilefull, thisfilename)-1);

% Add that path to the Matlab path.
addpath(thisfilepath);

%% Import all subpackages of Manopt

import manopt.*;
import manopt.tools.*;
import manopt.solvers.*;
import manopt.manifolds.*;

import manopt.solvers.linesearch.*;
import manopt.solvers.neldermead.*;
import manopt.solvers.pso.*;
import manopt.solvers.steepestdescent.*;
import manopt.solvers.conjugategradient.*;
import manopt.solvers.trustregions.*;

import manopt.manifolds.complexcircle.*;
import manopt.manifolds.euclidean.*;
import manopt.manifolds.fixedrank.*;
import manopt.manifolds.grassmann.*;
import manopt.manifolds.oblique.*;
import manopt.manifolds.rotations.*;
import manopt.manifolds.sphere.*;
import manopt.manifolds.stiefel.*;
import manopt.manifolds.symfixedrank.*;
