@echo off
@setlocal EnableDelayedExpansion

echo must install openclsdk,python27,git,cmake,vs 2015 for desktop (not for win10)
echo if download tar from https://github.com/willyd/caffe-builder/releases/ (see WindowsDownloadPrebuiltDependencies.cmake)fail, please copy it to build and delete libraries dir

for /f "delims=" %%t in ('python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"') do set py_path_str=%%t

set codepath=%~sdp0\..\..\

cd %codepath%
git clone https://github.com/dlfcn-win32/dlfcn-win32
cd dlfcn-win32
cmake -G "Visual Studio 14 2015 Win64" .
cmake --build . --config Release
cd %codepath%
git clone https://github.com/ptillet/isaac.git
cd isaac
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" ..
cmake --build . --config Release
cd %codepath%
::git clone https://github.com/01org/caffe.git
cd caffe
git checkout inference-optimize
git pull
git clone https://github.com/viennacl/viennacl-dev.git
set BUILD_PYTHON=1
set BUILD_PYTHON_LAYER=1
set USE_INTEL_SPATIAL=1
set USE_GREENTEA=1
set USE_ISAAC=1
set RUN_TESTS=0
set RUN_INSTALL=1
scripts\build_win.cmd

cd %codepath%
for /f  "tokens=1,2 delims==" %%b in (%codepath%\caffe\build\CMakeCache.txt) do (
	if "%%b"=="OPENCL_LIBRARIES:FILEPATH" set OPENCL_LIBRARIES=%%c
	if "%%b"=="PYTHON_LIBRARY:FILEPATH" set PYTHON_LIBRARY=%%c
)
set OPENCL_LIBRARIES=%OPENCL_LIBRARIES:/=\%
set PYTHON_LIBRARY=%PYTHON_LIBRARY:/=\%

if not exist "%codepath%\caffe\build\install\" (
	echo do not find caffe build
)else (
	:: copy lib and include
	copy /y %codepath%\dlfcn-win32\Release\dl.dll %codepath%\caffe\build\install\bin
	copy /y %codepath%\isaac\build\lib\Release\isaac.dll %codepath%\caffe\build\install\bin
	copy /y %codepath%\dlfcn-win32\Release\dl.dll %codepath%\caffe\build\install\python\caffe
	copy /y %codepath%\isaac\build\lib\Release\isaac.dll %codepath%\caffe\build\install\python\caffe

	copy /y %codepath%\caffe\build\libraries\lib\boost_python-vc140-mt-1_61.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\boost_system-vc140-mt-1_61.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\boost_thread-vc140-mt-1_61.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\boost_filesystem-vc140-mt-1_61.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\boost_regex-vc140-mt-1_61.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\boost_chrono-vc140-mt-1_61.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\boost_date_time-vc140-mt-1_61.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\boost_atomic-vc140-mt-1_61.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\glog.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\gflags.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\libprotobuf.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\caffehdf5_hl.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\caffehdf5.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\caffezlib.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\lmdb.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\leveldb.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\snappy_static.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\lib\libopenblas.dll.a %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\x64\vc14\lib\opencv_highgui310.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\x64\vc14\lib\opencv_videoio310.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\x64\vc14\lib\opencv_imgcodecs310.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\x64\vc14\lib\opencv_imgproc310.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\caffe\build\libraries\x64\vc14\lib\opencv_core310.lib %codepath%\caffe\build\install\lib
	copy /y %codepath%\isaac\build\lib\Release\isaac.lib %codepath%\caffe\build\install\lib

	copy /y %OPENCL_LIBRARIES% %codepath%\caffe\build\install\lib
	copy /y %PYTHON_LIBRARY% %codepath%\caffe\build\install\lib

	xcopy %codepath%\caffe\build\libraries\include %codepath%\caffe\build\install\include /s /h /c /y 
	move /y %codepath%\caffe\build\install\include\boost-1_61\boost %codepath%\caffe\build\install\include\boost 
	mkdir %codepath%\caffe\build\install\include\viennacl
	xcopy %codepath%\caffe\viennacl-dev\viennacl %codepath%\caffe\build\install\include\viennacl /s /h /c /y 
	mkdir  %codepath%\caffe\build\install\include\CL
	xcopy %codepath%\caffe\viennacl-dev\CL %codepath%\caffe\build\install\include\CL /s /h /c /y 
	mkdir  %codepath%\caffe\build\install\include\3rdparty
	xcopy %codepath%\caffe\include\3rdparty %codepath%\caffe\build\install\include\3rdparty /s /h /c /y 

	:: install python
	cd %py_path_str%\..\..\Scripts
	pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
	echo ###############################################
	echo copy caffe\build\install\python\caffe to  %py_path_str%\caffe
	cd %py_path_str%
	mkdir caffe
	xcopy %codepath%\caffe\build\install\python\caffe .\caffe /s /h /c /y 
	
	%codepath%\caffe\scripts\nuget\nuget-create.cmd
)
