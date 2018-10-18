% Script to facilitate the generation of a new release of Manopt.

fprintf('Release? Are your sure? CTRL+C if not. Any other key if yes.\n');
pause();

reproot = pwd(); % Manopt root directory -- no slash at the end
repreleases = '../manopt-releases/'; % relative to root (slash at the end)
repweb = '../manopt-web/'; % relative to root (slash at the end)

version_str = input('What version is this? (Between quotes. Example: ''1.0.42'') ');

% This has to be done manually for now.
fprintf('You must now manually edit these files for version names and date:\n');
fprintf('README.txt and manopt_version.m.\n');
fprintf('Today is: %s\n', date());
fprintf('Press any key AFTER you are done.\n');
edit auxiliaries/manopt_version.m;
edit auxiliaries/README.txt;
edit auxiliaries/CREDITS.txt;
pause();

cd(repreleases);
repthisversion = sprintf('Manopt_%s', version_str); % no slash at the end
mkdir(repthisversion);

cd(repthisversion);
mkdir manopt;

% We are now back at the root of the manopt repository
cd(reproot);

% Copy everything that's needed for a release.

targetrep = [repreleases, repthisversion, '/manopt/'];

copyfile('importmanopt.m', targetrep, 'f');
copyfile('examples', [targetrep, 'examples/'], 'f');
copyfile('manopt', [targetrep, 'manopt/'], 'f');
copyfile('auxiliaries', targetrep, 'f');

% Let's go and erase all *.asv files
cd(targetrep);
dirlist = strread(genpath(pwd()), '%s', 'delimiter', ';');
for subdir = dirlist'
    delete([subdir{1}, '/*.asv']);
end

% Move to the newly created folder in the new release folder
cd(reproot);
cd(repreleases);
cd(repthisversion);

% Compress all contents into a zip file in the web folder
zip([reproot, '/', repweb, 'downloads/', repthisversion, '.zip'], '*');

% Back to the root
cd(reproot);

% A heads up for reference generation
fprintf('About to generate the reference for the website. Press any key.\n');
pause();
cd reference;
generate_manopt_reference([repweb 'reference/']);

fprintf('Now: update the website !\n');
fprintf('Files concerned: download.html, downloads.html and tutorial.html\n');
cd(reproot);
cd(repweb);
edit download.html;
edit downloads.html;
edit tutorial.html;

% And finish home
cd(reproot);

