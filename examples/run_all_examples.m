clear all; %#ok<CLALL>
close all;
clc;

list = dir('*.m');

issues = {};

for k = 1 : numel(list)
    s = list(k).name;
    s = s(1:end-2); % remove '.m'
    switch s
        case mfilename() % don't run yourself
        case 'positive_definite_karcher_mean' % old name of another example
        otherwise
            try
                eval([s, ';']);
            catch
                issues{end+1} = s; %#ok<SAGROW>
            end
    end
end

if ~isempty(issues)
    warning('There were issues running the following scripts:');
    disp(issues);
else
    fprintf('\n\nNo particular issues detected.\n');
end
