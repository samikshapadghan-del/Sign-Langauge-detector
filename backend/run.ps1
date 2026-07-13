$ErrorActionPreference = "Stop"

$BackendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $BackendDir
$RootLauncher = Join-Path $ProjectRoot "run.ps1"

& $RootLauncher
