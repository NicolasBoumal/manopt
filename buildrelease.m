% Attempt at a script to facilitate the generation of a new release of
% manopt.

fprintf('Release? Are your sure? CTRL+C if not. Any other key if yes.\n');
pause();

version_str = input('What version is this? (between quotes. example: ''1.0.42'') ');

% This has to be done manually for now.
fprintf('You must now manually edit these files for version names and date:\n');
fprintf('README.txt and manopt_version.m.\n');
fprintf('Today is: %s\n', date());
fprintf('Press any key AFTER you are done.\n');
edit auxiliaries/manopt_version.m;
edit auxiliaries/README.txt;
pause();

cd releases;
rep = sprintf('Manopt_%s', version_str);
mkdir(rep);

cd(rep);
mkdir manopt;
cd ..;
cd ..;

% We are now back at the root of the manopt repository

% Copy everything that's needed for a release.

targetrep = ['releases/', rep, '/manopt/'];

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
cd ..;
cd ..;
cd ..;

cd releases;
cd(rep);
zip(['../../web/downloads/', rep, '.zip'], '*');
cd ..;
cd ..;

% A heads up for reference generation
fprintf('About to generate the reference for the website. Press any key.\n');
pause();
cd reference;
generate_manopt_reference;

fprintf('Now: update the website !\n');
fprintf('Files concerned: download.html, downloads.html and tutorial.html\n');
edit web/download.html;
edit web/downloads.html;
edit web/tutorial.html;
