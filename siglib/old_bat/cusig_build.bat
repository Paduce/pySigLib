@echo off

set SIGLIBDIR=C:\Users\Shmelev\source\repos\pySigLib\siglib
set VCINSTALLDIR="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
set NVCC_EXE="%CUDA_PATH%\bin\nvcc.exe"
set CL_EXE=%VCINSTALLDIR%\bin\HostX64\x64\cl.exe
set LINK_EXE=%VCINSTALLDIR%\bin\HostX64\x64\link.exe

@echo SIGLIBDIR=%SIGLIBDIR%
@echo VCINSTALLDIR=%VCINSTALLDIR%
@echo NVCC_EXE=%NVCC_EXE%
@echo CL_EXE=%CL_EXE%
@echo LINK_EXE=%LINK_EXE%


@echo off
REM set env variables for 64b c++ 
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

@echo INCLUDE=%INCLUDE%

REM set current dir
CD %SIGLIBDIR%\cusig


@echo build pch

md x64
cd x64
md Release

CD %SIGLIBDIR%\cusig

%CL_EXE% /c /I"%CUDA_PATH%\include" /Zi /nologo /W3 /WX- /diagnostics:column /sdl /O2 /Oi /GL /D NDEBUG /D CUSIG_EXPORTS /D _WINDOWS /D _USRDLL /D _WINDLL /D _UNICODE /D UNICODE /Gm- /EHsc /MT /GS /Gy /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline /std:c++20 /permissive- /Yc"cupch.h" /Fp"x64\Release\cusig.pch" /Fo"x64\Release\\" /Fd"x64\Release\vc143.pdb" /external:W3 /Gd /TP /FC /errorReport:prompt cupch.cpp

pause

@echo compile with cl.exe
%CL_EXE% /c /I"%CUDA_PATH%\include" /Zi /nologo /W3 /WX- /diagnostics:column /sdl /O2 /Oi /GL /D NDEBUG /D CUSIG_EXPORTS /D _WINDOWS /D _USRDLL /D _WINDLL /D _UNICODE /D UNICODE /Gm- /EHsc /MT /GS /Gy /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline /std:c++20 /permissive- /Yu"cupch.h" /Fp"x64\Release\cusig.pch" /Fo"x64\Release\\" /Fd"x64\Release\vc143.pdb" /external:W3 /Gd /TP /FC /errorReport:prompt cusig.cpp
%CL_EXE% /c /I"%CUDA_PATH%\include" /Zi /nologo /W3 /WX- /diagnostics:column /sdl /O2 /Oi /GL /D NDEBUG /D CUSIG_EXPORTS /D _WINDOWS /D _USRDLL /D _WINDLL /D _UNICODE /D UNICODE /Gm- /EHsc /MT /GS /Gy /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline /std:c++20 /permissive- /Yu"cupch.h" /Fp"x64\Release\cusig.pch" /Fo"x64\Release\\" /Fd"x64\Release\vc143.pdb" /external:W3 /Gd /TP /FC /errorReport:prompt dllmain.cpp





@echo compile cuda files with nvcc

pause

REM compile cuda files with nvcc
rem %CL_EXE% /E /nologo /showIncludes /TP /D__CUDACC__ /D__CUDACC_VER_MAJOR__=12 /D__CUDACC_VER_MINOR__=6 /DNDEBUG /DCUSIG_EXPORTS /D_WINDOWS /D_USRDLL /D_WINDLL /D_UNICODE /DUNICODE /I"%CUDA_PATH%\include" /I"%CUDA_PATH%\bin" /I"%CUDA_PATH%\include" /I. /FIcuda_runtime.h /c %SIGLIBDIR%\cusig\cuSigKernel.cu
rem pause

set CMD1=%NVCC_EXE% -gencode=arch=compute_52,code=\"sm_52,compute_52\" --use-local-env -ccbin "%VCINSTALLDIR%bin\HostX64\x64" -x cu -rdc=true  -I"%CUDA_PATH%\include" --keep-dir x64\Release  -maxrregcount=0   --machine 64 --compile -cudart static -lineinfo   -DNDEBUG -DCUSIG_EXPORTS -D_WINDOWS -D_USRDLL -D_WINDLL -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W3 /nologo /O2 /FS   /MT "  -o cuSigKernel.cu.obj   "cuSigKernel.cu"

echo CMD1=%CMD1%

pause

%CMD1%

pause

%NVCC_EXE% -gencode=arch=compute_52,code=\"sm_52,compute_52\" --use-local-env -ccbin "%VCINSTALLDIR%\bin\HostX64\x64" -x cu -rdc=true  -I"%CUDA_PATH%\include" -I"%CUDA_PATH%\include"     --keep-dir x64\Release  -maxrregcount=0   --machine 64 --compile -cudart static -lineinfo   -DNDEBUG -DCUSIG_EXPORTS -D_WINDOWS -D_USRDLL -D_WINDLL -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W3 /nologo /O2 /FS   /MT " -Xcompiler "/Fdx64\Release\vc143.pdb" -o %SIGLIBDIR%\cusig\x64\Release\vectoradd.cu.obj "%SIGLIBDIR%\cusig\vectoradd.cu"

%NVCC_EXE% -gencode=arch=compute_52,code=\"sm_52,compute_52\" --use-local-env -ccbin "%VCINSTALLDIR%\bin\HostX64\x64" -x cu -rdc=true  -I"%CUDA_PATH%\include" -I"%CUDA_PATH%\include"     --keep-dir x64\Release  -maxrregcount=0   --machine 64 --compile -cudart static -lineinfo   -DNDEBUG -DCUSIG_EXPORTS -D_WINDOWS -D_USRDLL -D_WINDLL -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W3 /nologo /O2 /FS   /MT " -Xcompiler "/Fdx64\Release\vc143.pdb" -o %SIGLIBDIR%\cusig\x64\Release\cuSigKernel.h.obj "%SIGLIBDIR%\cusig\cuSigKernel.h"

%NVCC_EXE% -gencode=arch=compute_52,code=\"sm_52,compute_52\" --use-local-env -ccbin "%VCINSTALLDIR%\bin\HostX64\x64" -x cu -rdc=true  -I"%CUDA_PATH%\include" -I"%CUDA_PATH%\include"     --keep-dir x64\Release  -maxrregcount=0   --machine 64 --compile -cudart static -lineinfo   -DNDEBUG -DCUSIG_EXPORTS -D_WINDOWS -D_USRDLL -D_WINDLL -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W3 /nologo /O2 /FS   /MT " -Xcompiler "/Fdx64\Release\vc143.pdb" -o %SIGLIBDIR%\cusig\x64\Release\vectoradd.h.obj "%SIGLIBDIR%\cusig\vectoradd.h"



pause

@echo ---------------------------------------------------------------------------------------
@echo link cuda obj files
REM link cuda obj files 
%NVCC_EXE% -dlink  -o x64\Release\cusig.device-link.obj -Xcompiler "/EHsc /W3 /nologo /O2   /MT " -Xcompiler "/Fdx64\Release\vc143.pdb" -L"%CUDA_PATH%\bin/crt" -L"%CUDA_PATH%\lib\x64" kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib cudart.lib cudadevrt.lib  -gencode=arch=compute_52,code=sm_52   %SIGLIBDIR%\cusig\x64\Release\cuSigKernel.h.obj %SIGLIBDIR%\cusig\x64\Release\vectoradd.h.obj %SIGLIBDIR%\cusig\x64\Release\cuSigKernel.cu.obj %SIGLIBDIR%\cusig\x64\Release\vectoradd.cu.obj

pause



@echo ---------------------------------------------------------------------------------------
@echo link.exe

REM link final 

CD %SIGLIBDIR%

md x64
cd x64
md Release

CD %SIGLIBDIR%\cusig


%LINK_EXE% /ERRORREPORT:PROMPT /OUT:"%SIGLIBDIR%\x64\Release\cusig.dll" /NOLOGO /LIBPATH:"%CUDA_PATH%\lib\x64" kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib cudart.lib cudadevrt.lib /MANIFEST /MANIFESTUAC:NO /manifest:embed /DEBUG /PDB:"%SIGLIBDIR%\x64\Release\cusig.pdb" /SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /LTCG:incremental /LTCGOUT:"x64\Release\cusig.iobj" /TLBID:1 /DYNAMICBASE /NXCOMPAT /IMPLIB:"%SIGLIBDIR%\x64\Release\cusig.lib" /MACHINE:X64 /DLL %SIGLIBDIR%\cusig\x64\Release\cuSigKernel.h.obj %SIGLIBDIR%\cusig\x64\Release\vectoradd.h.obj %SIGLIBDIR%\cusig\x64\Release\cuSigKernel.cu.obj %SIGLIBDIR%\cusig\x64\Release\vectoradd.cu.obj x64\Release\cusig.obj x64\Release\dllmain.obj  x64\Release\cupch.obj  "x64\Release\cusig.device-link.obj"

@echo ---------------------------------------------------------------------------------------

pause

