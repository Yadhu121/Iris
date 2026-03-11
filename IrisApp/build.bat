@echo off
title Build GestureControl.exe
echo.
echo  Building GestureControl WPF launcher...
echo.

dotnet --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] .NET SDK not found.
    echo  Download from: https://dotnet.microsoft.com/download
    pause
    exit /b 1
)

cd GestureControlApp

dotnet publish -c Release -r win-x64 --self-contained true ^
    /p:PublishSingleFile=true ^
    /p:IncludeNativeLibrariesForSelfExtract=true ^
    -o ..\dist

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Build failed. See output above.
    pause
    exit /b 1
)

cd ..

echo.
echo  Copying runtime files to dist\...
copy gesture_control.py dist\
copy setup.bat dist\
copy README.md dist\

echo.
echo   Build complete!
echo   Output in: dist\
echo   
echo   Zip the dist\ folder and host it on
echo   your website for download.
echo.
pause
