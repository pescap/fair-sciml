@ECHO OFF

set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "html" (
    %SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html
) else (
    echo "Usage: make.bat html"
)