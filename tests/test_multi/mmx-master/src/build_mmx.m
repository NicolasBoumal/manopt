function build_mmx(verbose)
% BUILD_MMX - compiles mmx() for different platforms and provides help
%            regarding compilation.
%
%  BUILD_MMX will try to compile, in this order, 3 different builds of mmx:
%  mmx_mkl_single    - linked to Intel's single-threaded MKL library (usually fastest)
%  mmx_mkl_multi     - linked to the multithreaded BLAS/LAPACK libraries that come
%                      with Matlab.
%  mmx_naive         - does not link to anything, uses simple C-loops.
%
%  The first time BUILD_MMX succeeds, it will compile again to 'mmx', so
%  that the mex-file mmx should be the fastest possible build on your
%  system.
%
%  BUILD_MMX has been tested on Win32, Win64, OSX, Linux 64
%

% %% FOR LINUX OR MAC SYSTEMS:
% 
% To properly link to Intel's MKL, user needs to repackage their libraries 
% into one single statically linked library. The instructions are as
% follows:
%
%
% Download Intel MKL for Linux here:
% http://software.intel.com/en-us/articles/non-commercial-software-download/
%
% Donwload Intel MKL for Mac here:
% https://registrationcenter.intel.com/RegCenter/AutoGen.aspx?ProductID=1518&AccountID=&EmailID=&ProgramID=&RequestDt=&rm=EVAL&lang=
%
% The Default installation directory for both Linux and Mac will be
% /opt/intel/
% with the MKL libraries in /opt/intel/mkl
%
% %% To build needed static Library
%    assuming default installation directory
%
% Run the following commands in Linux/Mac terminal:
%
% sudo -s
% cd /opt/intel/mkl/tools/builder
% cat blas_example_list > blas_lapack_list
% cat lapack_example_list >> blas_lapack_list
%
% For Linux 64 bit:
% make libintel64 interface=ilp64 export=blas_lapack_list name=libsingle_mkl_ilp64 threading=sequential
% For Linux 32 bit:
% make libia32 interface=lp64 export=blas_lapack_list name=libsingle_mkl_32 threading=sequential
%
% For Mac:
% make libuni interface=ilp64 export=blas_lapack_list name=libsingle_mkl_ilp64 threading=sequential
%
% A new libsingle_mkl_ilp64.so, libsingle_mkl_32.so, or 
% libsingle_mkl_ilp64.dylib will appear.
% This needs to be copied to Matlab's external libraries directory.
%
% For Mac:
% cp libsingle_mkl_ilp64* MATLAB_ROOT/extern/lib/maci64
%
% For Linux 64 bit:
% cp libsingle_mkl_ilp64* MATLAB_ROOT/extern/lib/glnxa64
% For Linux 32 bit:
% cp libsingle_mkl_32* MATLAB_ROOT/extern/lib/glnx86
%
% Where MATLAB_ROOT is the installation directory of your Matlab.


if nargin == 0
   verbose = false;
end

clc

build_names  = {'mmx_mkl_single', 'mmx_mkl_multi','mmx_naive'};

built_mmx   = false;

arch        = computer('arch');

for b = 1:3
   name = build_names{b};
   
   [link, define]  = deal({});
   [inc_dir, link_dir, Cflags, Lflags]  = deal('');
   
   switch arch
      case {'win64','win32'}
         switch name
            case 'mmx_naive'
               define   = {'WIN_SYSTEM'};
               
            case 'mmx_mkl_multi'
               root     = matlabroot;
               if strcmp(arch,'win32')
                  inc_dir  = [root '\extern\lib\win32\microsoft'];
               else
                  inc_dir  = [root '\extern\lib\win64\microsoft'];
               end
               link     = {'libmwblas','libmwlapack'};
               define   = {'WIN_SYSTEM','USE_BLAS'};
               
            case 'mmx_mkl_single'
               root     = 'C:\Program Files (x86)\Intel\Composer XE 2011 SP1\mkl';
               inc_dir  = [root '\include'];
               if strcmp(arch,'win32')
                  link_dir  = [root '\lib\ia32'];
                  link     = {'mkl_intel_c','mkl_sequential','mkl_core'};
                  define   = {'WIN_SYSTEM','USE_BLAS','MKL_32'};
               else
                  link_dir  = [root '\lib\intel64'];
                  link     = {'mkl_intel_ilp64','mkl_sequential','mkl_core'};
                  define   = {'WIN_SYSTEM','USE_BLAS','MKL_ILP64'};
               end
         end
      case {'glnxa64','glnx86'}
         switch name
            case 'mmx_naive'
               link     = {'pthread'};
               define   = {'UNIX_SYSTEM'};
            case 'mmx_mkl_multi'
               if strcmp(arch,'glnx86')
               inc_dir  = [matlabroot '/extern/lib/glnx86'];
               else
               inc_dir  = [matlabroot '/extern/lib/glnxa64'];
               end
               link     = {'mwblas','mwlapack','pthread'};
               define   = {'UNIX_SYSTEM','USE_BLAS'};
            case 'mmx_mkl_single'
               root = '/opt/intel/mkl';
               inc_dir  = [ root '/include'];
               if strcmp(arch,'glnx86')
                link_dir  = [matlabroot '/extern/lib/glnx86'];
                link     = {'single_mkl_32','pthread'};
                define   = {'UNIX_SYSTEM', 'USE_BLAS', 'MKL_32'};
               else
                link_dir  = [matlabroot '/extern/lib/glnxa64'];
                link     = {'small_mkl_ilp64','pthread'};
                define   = {'UNIX_SYSTEM', 'USE_BLAS', 'MKL_ILP64'};
               end
         end
      case {'maci64'}
         switch name
            case 'mmx_naive'
               link     = {'pthread'};
               define   = {'UNIX_SYSTEM'};
               
            case 'mmx_mkl_multi'
               root     = matlabroot;
               inc_dir  = [root '/extern/lib/maci64'];
               link     = {'mwblas','mwlapack','pthread'};
               define   = {'UNIX_SYSTEM','USE_BLAS'};
               
            case 'mmx_mkl_single'
               root     = '/opt/intel/mkl';
               inc_dir  = [ root '/include'];
               link_dir = [matlabroot '/extern/lib/maci64'];
               link     = {'single_mkl_ilp64','pthread'};
               %link     = {'small_mkl_ilp64','pthread'};
               define   = {'UNIX_SYSTEM', 'USE_BLAS', 'MKL_ILP64'};
         end
         
      otherwise
         error unsupported_architecture
   end
   
   if ~isempty(link_dir)
      if strcmp(arch,'glnxa64') || strcmp(arch,'maci64')
         L_dir  = {['LDFLAGS="\$LDFLAGS -L' link_dir  ' ' Lflags '"']};
      else
         L_dir  = {['-L' link_dir]};
      end
   else
      L_dir  = {};
   end
   
   if ~isempty(inc_dir)
      if strcmp(arch,'glnxa64') || strcmp(arch,'maci64')
         I_dir  = {['CXXFLAGS="\$CXXFLAGS -I' inc_dir ' ' Cflags '"']};
      else
         I_dir  = {['-I' inc_dir]};
      end
   else
      I_dir  = {};
   end
   
   prefix   = @(pref,str_array) cellfun(@(x)[pref x],str_array,'UniformOutput',0);
   l_link   = prefix('-l',link);
   D_define = prefix('-D',define);
   
   if verbose
      verb  = {'-v'};
   else
      verb  = {};
   end
   
   try
      check_dir(link_dir, link)
      check_dir(inc_dir)
      clear(name)
      command = {verb{:}, I_dir{:}, L_dir{:}, l_link{:}, D_define{:}}; %#ok<*CCAT>
      fprintf('==========\nTrying to compile ''%s'', using \n',name);
      fprintf('%s, ',command{:})
      fprintf('\n')
      mex(command{:}, '-output', name, 'mmx.cpp');
      fprintf('Compilation of ''%s'' succeeded.\n',name);
      if ~built_mmx
         fprintf('Compiling again to ''mmx'' target using ''%s'' build.\n',name);
         mex(command{:}, '-output','mmx','mmx.cpp');
         built_mmx = true;
      end
   catch err
      fprintf('Compilation of ''%s'' failed with error:\n%s\n',name,err.message);
   end
end

function check_dir(dir,files)
if ~isempty(dir)
   here = cd(dir);
   if nargin == 2
      for i = 1:size(files)
         if isempty(ls(['*' files{i} '.*']))
            cd(here);
            error('could not find file %s', files{i});
         end
      end
   end
   cd(here);
end



