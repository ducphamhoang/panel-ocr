@echo off
setlocal EnableExtensions

rem panel-ocr Windows first-time setup.
rem
rem This is packaging-only convenience, not app behavior: it writes the SAME config.toml a
rem user could hand-author (spec section 16.12 item 21 / 16.33), at the same path the app
rem already looks for one (%APPDATA%\panel-ocr\config.toml). Nothing here is read by the
rem Rust binary directly - it only ever consults config.toml and --cache-dir, both of which
rem already existed before this script did.

set "DEFAULT_DIR=%LOCALAPPDATA%\panel-ocr"
set "CONFIG_DIR=%APPDATA%\panel-ocr"
set "CONFIG_FILE=%CONFIG_DIR%\config.toml"
set "EXE=%~dp0panel-ocr.exe"

echo panel-ocr first-time setup
echo.

if not exist "%EXE%" (
    echo Could not find panel-ocr.exe next to this script ^(expected at "%EXE%"^).
    echo Run setup.bat from inside the extracted release folder.
    echo.
    pause
    exit /b 1
)

if exist "%CONFIG_FILE%" (
    echo A config already exists at:
    echo   %CONFIG_FILE%
    echo.
    set /p OVERWRITE="Overwrite it? [y/N]: "
    rem First character only, not full-string equality: `set /p` can leave a trailing
    rem artifact (space, CR) after piped input, same class of quirk as the CACHE_DIR trim
    rem below, and a yes/no answer only ever needs its first character checked anyway.
    if /i not "%OVERWRITE:~0,1%"=="y" (
        echo Leaving the existing config untouched. Nothing changed.
        pause
        exit /b 0
    )
    echo.
)

echo Where should panel-ocr store its cache and downloaded models?
echo Press Enter to use the default:
echo   %DEFAULT_DIR%
set /p CACHE_DIR="Path: "
rem `set /p` fed from a pipe (as this script's own CI smoke test does, and as some terminal
rem emulators do) can yield a single space rather than a true empty string for a blank line -
rem confirmed by this exact failure on windows-latest (release run 30912573913). Trim leading
rem whitespace before the empty check, so both real keyboard Enter and a piped blank line take
rem the same default-substitution path.
for /f "tokens=* delims= " %%A in ("%CACHE_DIR%") do set "CACHE_DIR=%%A"
if "%CACHE_DIR%"=="" set "CACHE_DIR=%DEFAULT_DIR%"

if not exist "%CACHE_DIR%" (
    mkdir "%CACHE_DIR%" 2>nul
    if not exist "%CACHE_DIR%" (
        echo.
        echo Could not create "%CACHE_DIR%". Check the path and try again.
        pause
        exit /b 1
    )
)

if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%" 2>nul

rem TOML literal strings ('...') take backslashes literally, so a raw Windows path needs no
rem escaping here - confirmed against pc-config's real parser before this script was written.
(
    echo # Written by setup.bat on %DATE% %TIME%
    echo # Edit cache_dir any time to change where panel-ocr stores models and cache data.
    echo # See spec section 16.12 item 21 / 16.33 for the full precedence rules ^(--cache-dir
    echo # on the command line always wins over this file^).
    echo cache_dir = '%CACHE_DIR%'
) > "%CONFIG_FILE%"

echo.
echo Wrote %CONFIG_FILE%
echo   cache_dir = %CACHE_DIR%
echo.

set /p DOWNLOAD="Download the ONNX model weights now (~90 MB)? [Y/n]: "
echo DEBUG_DOWNLOAD_RAW=[%DOWNLOAD%]
rem First character only - see the OVERWRITE check above for why. Confirmed necessary by
rem this exact failure mode on windows-latest (release run 30913311788): a trailing artifact
rem after piped "n" made the full-string comparison miss, and the script downloaded anyway.
if /i "%DOWNLOAD:~0,1%"=="n" (
    echo.
    echo Skipped. Run this later to fetch the models:
    echo   "%EXE%" models download
    echo.
    pause
    exit /b 0
)

echo.
echo Downloading model weights to %CACHE_DIR%\models ...
echo.
"%EXE%" models download
set "DOWNLOAD_RESULT=%ERRORLEVEL%"

echo.
if not "%DOWNLOAD_RESULT%"=="0" (
    echo Download failed ^(exit code %DOWNLOAD_RESULT%^). Check your network connection, then retry:
    echo   "%EXE%" models download
) else (
    echo Setup complete. Try:
    echo   "%EXE%" clean --detector onnx some-page.png
)

echo.
pause
