// =======================
// preprocessor directives
// =======================
#include "mex.h"
#include "string.h"
#include <ctype.h>

#ifdef UNIX_SYSTEM
   #include <unistd.h>
   #include <pthread.h>
#endif

#ifdef WIN_SYSTEM
   #include "windows.h"
#endif

#include <math.h>

#if defined(USE_BLAS)
   #if defined(MKL_ILP64) or defined(MKL_32)
      #include "mkl_blas.h"
      #include "mkl_lapack.h"
      #define ptrdiff_t MKL_INT
   #else 
      #ifdef UNIX_SYSTEM
        #include "blas.h"
        #include "lapack.h"
      #else
        #include "my_blas.h"
      #endif
   #endif
#endif

#define MAX_THREAD 64

// list of possible values for PARTASK
// positive values are matrix oprations
#define MATMUL 1
#define SQUARE 2
#define CHOL   3
#define BSLASH 4
// negative values are binary-elementwise functions (like bsxfun)
// (removed)
        
// function declarations
#include "matrix_fun.c"

//#define DEBUG

// ================
// global variables
// ================

// thread related
static bool    INITIALIZED = false;
static int     SCHEDULE[MAX_THREAD][2];   //start and stop index for each thread
static int     NTHREAD = 0;

#ifdef WIN_SYSTEM
static HANDLE  THREAD[MAX_THREAD];
static HANDLE  TSTART[MAX_THREAD];
static HANDLE  TDONE[MAX_THREAD];
#endif


// computation related
int   PARTASK;
double *A, *B, *C, *WORK, *C2;
int a1, a2, b1 ,b2, c1, c2, strideA, strideB, strideC, strideW, strideC2;
int *PAIRS = NULL;
ptrdiff_t *iScratch = NULL;
bool  BSX;
//bool USED_DGELSY = false;
char MODIFY[2];

// ================================
// teval() is called by each thread
// ================================
#ifdef WIN_SYSTEM
DWORD __stdcall teval(void* pn) 
#else
void* teval(void* pn) 
#endif
{
   // get thread number
   int i, n = *(int*)pn;
   ptrdiff_t *Si;
   double *Ai, *Bi, *Wi;

#ifdef WIN_SYSTEM
   while(1){ // thread will be terminated externally
      WaitForSingleObject(TSTART[n], INFINITE); // wait for start signal
#endif
      // loop over data
      for( i=SCHEDULE[n][0]; i<SCHEDULE[n][1]; i++ ){
         // pointers to scheduled data
         if (BSX){      //singleton expansion
            Ai = A + strideA*PAIRS[2*i];
            Bi = B + strideB*PAIRS[2*i+1];
         } else {
            Ai = A + strideA*i;
            Bi = B + strideB*i;
         }
         // excecute the task
         switch ( PARTASK ) {
            case MATMUL:
               mulMatMat(C + strideC*i, Ai, Bi , a1, a2, b1, b2, MODIFY);
               //mexPrintf("%d strideC: %d A %f, B %f, C %f\n", i, strideC, Ai[0], Bi[0], *(C+strideC*i));
               break;
            case SQUARE:
               squareMatMat(C + strideC*i, Ai, Bi , a1, a2, b1, b2, MODIFY);
               break;
            case CHOL:
               chol(C + strideC*i, Ai, a1);
               break;
            case BSLASH:
               if (iScratch != NULL && WORK != NULL) {
                  Wi = WORK + strideW*i;
                  Si = iScratch + a2*i;
                  solve(C + strideC*i, Ai, Bi , a1, a2, b1, b2, MODIFY, Wi, strideW, Si);
               }
               break;               
         }
      }
#ifdef WIN_SYSTEM
      //signal that thread is finished
      SetEvent(TDONE[n]);
   }
#endif

   return 0;
}


// =============
// mexFunction()
// =============
void mexFunction(int n_out, mxArray *p_out[], int n_in, const mxArray *p_in[])
{
   mwSize  Andim, Bndim, Cndim;
   mwSize *Adims, *Bdims, *Adims_full, *Bdims_full, *Cdims, *idx;
   mxArray *tArray = NULL;
   char chr;
   int i, j, k, nt, N, iA, iB, iC, iC2, c12 = 0;
   static int tnum[MAX_THREAD];

   // no input: print documentation
   if( n_in==0 ) {
      mexPrintf(  "=======================================================\n"
            "===  mmx(): fast, multithreaded, n-D multiplication ===\n"
            "Basic usage:\nThe command   C = mmx('mult',A,B);\n"
            "is equivalent to the matlab loop\n"
            " for i=1:N\n    C(:,:,i) = A(:,:,i)*B(:,:,i);\n end\n"
            "===== Type 'help mmx' for detailed information. =======\n"
            "=======================================================\n");   
   }
   // ===================
   // threading machinery
   // ===================

   // single input mmx(nt): set number of threads
   if( n_in==1 ){
      if((!mxIsNumeric(p_in[0]))||(mxGetN(p_in[0])!=1)||(mxGetM(p_in[0])!=1))
         mexErrMsgTxt("A single scalar input specifies the desired thread count. Type 'help mmx' for more info.");
      nt = (int)mxGetScalar(p_in[0]);
   }
   else if (n_in==0 || !INITIALIZED) {
#ifdef WIN_SYSTEM
      SYSTEM_INFO sysinfo;
      GetSystemInfo( &sysinfo );
      nt = sysinfo.dwNumberOfProcessors;
#else
      nt = sysconf(_SC_NPROCESSORS_ONLN);
#endif
   }
   else {
      nt = NTHREAD;
   }

   // if necessary, clear threads
   if ((nt==0) || (INITIALIZED && (nt!=NTHREAD))){
#ifdef WIN_SYSTEM
      mexPrintf("Clearing threads.\n");
      for( i=0; i<NTHREAD; i++ ) {
         TerminateThread(THREAD[i], 0);
         CloseHandle(THREAD[i]);
         CloseHandle(TSTART[i]);
         CloseHandle(TDONE[i]);   
      }
#endif
      NTHREAD     = 0;
      INITIALIZED = false;      
   }

   // start threads
   if( !INITIALIZED && nt  ) {
      NTHREAD = (MAX_THREAD <= nt) ? MAX_THREAD : nt; // set global NTHREAD
      // create events and threads
      for( i=0; i<NTHREAD; i++ ) {
         tnum[i] = i;   // set tnum so CreateThread won't access i
         //mexPrintf("tnum[%d]: %d\n", i, tnum[i]);
      }
#ifdef WIN_SYSTEM
      for( i=0; i<NTHREAD; i++ ) {
         TSTART[i]   = CreateEvent(0, FALSE, FALSE, 0);
         TDONE[i]    = CreateEvent(0, TRUE, FALSE, 0);
         THREAD[i]   = CreateThread(NULL, 0, teval, (void*)(tnum+i), 0, 0);
      }
#endif
      INITIALIZED = true;
      if (n_in == 1) {//print this line only in single-input mode
         mexPrintf("%d threads prepared.\n", NTHREAD);
      }
   }

   // just getting help or setting the thread count, exit now
   if( n_in < 2 ) {
      return;   
   }

   // ==============
   // process inputs
   // ==============   

   // not enough inputs
   if( n_in == 2 ) {
      mexErrMsgTxt("Two is an invalid number of inputs.");
   }

   if(mxGetClassID(p_in[0]) != mxCHAR_CLASS) {
      mexErrMsgTxt("First argument must be a command. Type mmx() for more help.");
   }
   char *commandStr = mxArrayToString(p_in[0]);

   // process commands
   PARTASK = 0;
   switch (toupper(commandStr[0])){
      case 'M':
         PARTASK = MATMUL;
         break;
      case 'S':
         PARTASK = SQUARE; 
         break;
      case 'C':
         PARTASK = CHOL;
         break;
      case 'B':
         PARTASK = BSLASH;
#ifndef USE_BLAS
         mexErrMsgTxt("Recompile and link to BLAS to enable 'backslash' support");
#endif
         break;
      default:
         mexErrMsgTxt("Unknown command.");
   } 
   mxFree(commandStr);

   // type check
   if ( (!mxIsDouble(p_in[1])) || (!mxIsDouble(p_in[2])) ) {
      mexErrMsgTxt("Only inputs of type 'double' are supported.");
   }

   // get a1, a2, b1, b2
   A     = mxGetPr(p_in[1]);
   Andim = mxGetNumberOfDimensions(p_in[1]);
   Adims = (mwSize *) mxGetDimensions(p_in[1]);
   a1    = Adims[0];
   a2    = Adims[1];    

   B     = mxGetPr(p_in[2]);
   Bndim = mxGetNumberOfDimensions(p_in[2]);
   Bdims = (mwSize *) mxGetDimensions(p_in[2]);
   b1    = Bdims[0];
   b2    = Bdims[1]; 

   // modifiers
   MODIFY[0] = MODIFY[1] = 'N';
   if( n_in > 3 ){
      if(mxGetClassID(p_in[3]) != mxCHAR_CLASS)
         mexErrMsgTxt("Fourth argument is a modifier string. Type 'help mmx'.");
      char *modifierStr = mxArrayToString(p_in[3]);
      for( i=0; i<2; i++ ){
         chr = toupper(modifierStr[i]);
         switch ( PARTASK ){
            case MATMUL:
            case SQUARE:
               if ((chr == 'N')||(chr == 'T')||(chr == 'S'))
                  MODIFY[i] = chr;
               else if(chr!='\0')
                  mexErrMsgTxt("Unknown modifier.");
               break;
            case BSLASH:
               if ((chr == 'L')||(chr == 'U')||(chr == 'P'))
                  MODIFY[i] = chr;
               else if(chr!='\0')
                  mexErrMsgTxt("Unknown modifier for command BACKSLASH.");  
               break;          
         }
      }
      mxFree(modifierStr);
   }         

   // ================
   // dimension checks
   // ================  
   switch ( PARTASK ) {
      case MATMUL:
         if ( (MODIFY[0] == 'N' || MODIFY[0] == 'S') && (MODIFY[1] == 'N' || MODIFY[1] == 'S' ) && (a2 != b1) )
            mexErrMsgTxt("size(A,2) == size(B,1) should be true.");
         if ( (MODIFY[0] == 'T') && (MODIFY[1] == 'N') && (a1 != b1) )
            mexErrMsgTxt("size(A,1) == size(B,1) should be true.");
         if ( (MODIFY[0] == 'N') && (MODIFY[1] == 'T') && (a2 != b2) )
            mexErrMsgTxt("size(A,2) == size(B,2) should be true.");
         if ( (MODIFY[0] == 'T') && (MODIFY[1] == 'T') && (a1 != b2) )
            mexErrMsgTxt("size(A,1) == size(B,2) should be true.");      
         break;
      case SQUARE:
         if ((b1 !=0) && (b2 != 0))
            if ( (a1 != b1) || (a2 != b2) )
               mexErrMsgTxt("For SQUARE size(A,1)==size(B,1) and size(A,2)==size(B,2) should be true."); 
         break;    
      case CHOL:
         if (a1 != a2)
            mexErrMsgTxt("For CHOL size(A,1) == size(A,2) should be true."); 
         break; 
      case BSLASH:
         if (a1 != b1)
            mexErrMsgTxt("For BACKSLASH size(A,1) == size(B,1) should be true.");         
         if ( ((MODIFY[0] == 'L')||(MODIFY[0] == 'U')||(MODIFY[0] == 'P')) && (a1 != a2) )
            mexErrMsgTxt("For BACKSLASH size(A,1) == size(B,1) should be true."); 
         break;          
   } 

   // ===============
   // process outputs
   // =============== 

   Cndim    = (Andim > Bndim) ? Andim : Bndim;
   Cndim    = (Cndim > 3) ? Cndim : 3;
   Cdims    = (mwSize *) mxMalloc( Cndim * sizeof(mwSize) );
   idx      = (mwSize *) mxMalloc( Cndim * sizeof(mwSize) );

   // set Cdims[0,1]
   switch ( PARTASK ){
      case MATMUL:
         c1 = (MODIFY[0] == 'N' || MODIFY[0] == 'S') ? a1 : a2;
         c2 = (MODIFY[1] == 'N' || MODIFY[1] == 'S') ? b2 : b1;         
         break;
      case SQUARE:
         c2 = c1 = (MODIFY[0] == 'N') ? a1 : a2;
         break;
      case CHOL:
         c2 = c1 = a1;
         break;
      case BSLASH:
         c1  = (a1>a2) ? a1 : a2; // if a1>a2, overallocate rows for dgelsy's in-place shenanigans
         c12 = (a1>a2) ? a2 : 0;  // c12 saves the correct row count for C, we'll use it later to truncate
         c2  = b2;  
   }
   Cdims[0] = c1;
   Cdims[1] = c2; 


   // Adims_full and Bdims_full pad Adims and Bdims with 1s, if necessary
   Adims_full = (mwSize *) mxMalloc( Cndim * sizeof(mwSize) );
   Bdims_full = (mwSize *) mxMalloc( Cndim * sizeof(mwSize) );  

   // get Cdims and check singleton dimensions
   for( i=0; i<Cndim; i++ ) {
      Adims_full[i] = (i < Andim) ? Adims[i] : 1; 
      Bdims_full[i] = (i < Bndim) ? Bdims[i] : 1;
      if (i > 1){//check singleton-expanded dimensions
         Cdims[i] = (Adims_full[i] > Bdims_full[i]) ? Adims_full[i] : Bdims_full[i];
         if ( ( Adims_full[i]!=1 ) && ( Bdims_full[i]!=1 ) && ( Adims_full[i]!=Bdims_full[i] )  ){
            mexErrMsgTxt("Non-singleton dimensions of the two input arrays must match each other.");
         }         
      }
   }


   // stride sizes
   strideA    = a1*a2;
   strideB    = b1*b2;
   strideC    = c1*c2;

   // N is the total number of matrix operations
   N  = 1;
   for( i=2; i<Cndim; i++ ) {
      N *= Cdims[i];
   }

   // if one of the output dimensions is 0 we're done, goodbye
   if ( Cdims[0]*Cdims[1]*N == 0 ) {
      return;
   }

   // =====================================
   // compute pairs for singleton expansion
   // =====================================

   // check if singleton expansion be be avoided
   BSX   = false;
   if ( (b1 != 0) && (b2 != 0) ) {
      for( j=2; j<Cndim; j++ ) {
         if (Adims_full[j] != Bdims_full[j]) {
            BSX = true;
         }
      }
   }

   if (BSX) {
      // initialze idx
      for( j=2; j<Cndim; j++ ) {
         idx[j] = 0;
      }

      // init PAIRS
      PAIRS = (int *) mxMalloc( 2 * N * sizeof(int) );   
      PAIRS[0] = PAIRS[1] = 0;

      // compute PAIRS
      // (is there a fast way to do this inside the threads ???)
      for( i=1; i<N; i++ ){
         // idx = ind2sub(size(C), i) in C-style indexing
         idx[2]++;
         for( j=2; j<Cndim; j++ ) {
            if (idx[j] > Cdims[j]-1){
               idx[j] = 0;
               idx[j+1]++;
            }
         }
         // {iA,iB} = sub2ind(size({A,B}), idx) while ignoring singletons
         iA = iB = 0;
         for( j=Cndim-1; j>1; j-- ){
            if (Adims_full[j] > 1)  iA = iA*Adims_full[j] + idx[j];
            if (Bdims_full[j] > 1)  iB = iB*Bdims_full[j] + idx[j];         
         }
         PAIRS[2*i]     = iA;
         PAIRS[2*i+1]   = iB;      
      }
#ifdef DEBUG
      for( i=0; i<N; i++ ) {
         mexPrintf("%4d %4d %4d\n", i, PAIRS[2*i], PAIRS[2*i+1]);   
      }
#endif       
   }

   // =============================================
   // extra memory allocations for LAPACK functions
   // ============================================= 
#ifdef USE_BLAS
   if (PARTASK == BSLASH){
      //USED_DGELSY=false;
      if ((MODIFY[0] != 'L') && (MODIFY[0] != 'U') && (BSX))
         mexErrMsgTxt("Singleton expansion is not supported for LAPACK-based BACKSLASH.");
      switch (MODIFY[0]) {
         case 'P': // positive definite
            strideW  = strideA;
            tArray   = mxDuplicateArray(p_in[1]);
            WORK     = mxGetPr(tArray);
            break;
         default: // general, use LU (dgesv) or QR (dgelsy)
            if (a1 == a2) { // A is square
               strideW  = strideA;
               tArray   = mxDuplicateArray(p_in[1]);
               WORK     = mxGetPr(tArray);
               iScratch = (ptrdiff_t *) mxMalloc( N * a2 * sizeof(ptrdiff_t));
            } else { // A is not square, ask dgelsy how much scratch memory it needs
               //USED_DGELSY = true;
               double rcond = 0.000000001, worksize[10];
               ptrdiff_t rank;
               ptrdiff_t info, m_one=-1;
               ptrdiff_t a10 = a1, b20 = b2, a20=a2;
               ptrdiff_t b10= (a1>a2) ? a1 : a2;  
               dgelsy(&a10, &a20, &b20, A, &a10, B, &b10,
                     iScratch, &rcond, &rank, 
                     worksize, &m_one, &info);

               if (info != 0) {
                  mexPrintf("LAPACK memory allocation query failed.\n"); 
               }

               iScratch = (ptrdiff_t *) mxMalloc( N * a2 * sizeof(ptrdiff_t));
               strideW  = (int) worksize[0];
               WORK     = (double *) mxMalloc( N * strideW * sizeof(double) );

#ifdef DEBUG
               mexPrintf("mem required %f, query info = %d\n", WORK[0], info); 
#endif
               // duplicate A so it doesn't get corrupted
               tArray   = mxDuplicateArray(p_in[1]);
               A        = mxGetPr(tArray);
            }
      }  
   }   
#endif


   // allocate C
   if ((PARTASK == BSLASH) && (a1==c1)) {// initialize C=B for in-place square BACKSLASH
      p_out[0] = mxDuplicateArray(p_in[2]);
   }
   else {
      p_out[0] = mxCreateNumericArray(Cndim, Cdims, mxDOUBLE_CLASS, mxREAL);
   }
   n_out = 1;
   C  = mxGetPr(p_out[0]);

   // ==================================
   // make schedule, run threads, finish
   // ==================================   

   // set SCHEDULE
   int blksz = N/NTHREAD;
   int extra = N - blksz*NTHREAD;
   //mexPrintf("matrix_ops: %d block: %d extra: %d\n", N, blksz, extra);
   for( i=0; i<NTHREAD; i++ ) {
      SCHEDULE[i][0] = ((i>0) ? SCHEDULE[i-1][1] : 0);
      SCHEDULE[i][1] = SCHEDULE[i][0] + (blksz + (i<extra));
      //mexPrintf("SCHEDULE[%d] %d, %d\n", i, SCHEDULE[i][0], SCHEDULE[i][1]);
   }

   // signal threads to start
#ifdef WIN_SYSTEM
   for( i=0; i<NTHREAD; i++ ) {
      SetEvent(TSTART[i]);
   }

   //  wait for all threads to finish
   WaitForMultipleObjects(NTHREAD, TDONE, TRUE, INFINITE);

   // reset TDONE events
   for( i=0; i<NTHREAD; i++ ) {
      ResetEvent(TDONE[i]);
   }
#else

   pthread_t p_threads[NTHREAD];

   for( i=0; i<NTHREAD; i++ ) {
      if (pthread_create(&p_threads[i],
               NULL, teval, (void*)(tnum+i)) != 0) {
         mexPrintf("Could not create thread %d\n", i);
      }
   }

   for( i=0; i<NTHREAD; i++ )
   {
      if (pthread_join(p_threads[i], NULL) != 0) {
         mexPrintf("Could not join thread %d\n", i);    
      }
   }
#endif

   // if C was over-allocated, chop off the extra rows
   if (c12) {
      Cdims[0] = c12;
      //mxDestroyArray(p_out[0]);
      p_out[0] = mxCreateNumericArray(Cndim, Cdims, mxDOUBLE_CLASS, mxREAL);
      C2       = mxGetPr(p_out[0]);
      strideC2 = c12*c2;
      for( i=0; i<N; i++ ) {
         for( j=0; j<c2; j++ ){
            iC    = i*strideC+j*c1;
            iC2   = i*strideC2+j*c12;
            for( k=0; k<c12; k++ ) {
               C2[iC2+k] = C[iC+k];
            }
         }
      }
   } 

   mxFree(Adims_full);
   mxFree(Bdims_full);
   if (BSX) {
      mxFree(PAIRS);
   }
   mxFree(Cdims);
   mxFree(idx);

   if (tArray != NULL) {
      mxDestroyArray(tArray);
   }
}

