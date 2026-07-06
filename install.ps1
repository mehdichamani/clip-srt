<#
.SYNOPSIS
  Sets up a Python virtual environment and installs project dependencies on Windows (PowerShell).

.PARAMETER ProxyUrl
  Optional HTTP/HTTPS proxy URL to use for downloads (e.g. http://127.0.0.1:2080). SOCKS proxies are ignored.

USAGE
  .\install.ps1
  .\install.ps1 -ProxyUrl http://127.0.0.1:2080
#>

param(
    [string]$ProxyUrl = ""
)

Write-Host "== clip-srt Windows installer =="

try {
    Write-Host "Setting execution policy for this session..."
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force | Out-Null

    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "Python is not in PATH. Install Python 3.10+ and re-run this script."
        exit 1
    }

    $venvPath = ".venv"
    if (-not (Test-Path $venvPath)) {
        Write-Host "Creating virtual environment at $venvPath..."
        python -m venv $venvPath
    } else {
        Write-Host "Virtual environment already exists at $venvPath"
    }

    Write-Host "Activating virtual environment..."
    .\$venvPath\Scripts\Activate.ps1

    Write-Host "Upgrading pip, setuptools and wheel..."
    python -m pip install --upgrade pip setuptools wheel

    if ($ProxyUrl) {
        if ($ProxyUrl.StartsWith('http://') -or $ProxyUrl.StartsWith('https://')) {
            Write-Host "Using HTTP proxy: $ProxyUrl"
            $env:PROXY_URL = $ProxyUrl
            $env:HTTP_PROXY = $ProxyUrl
            $env:HTTPS_PROXY = $ProxyUrl
        } else {
            Write-Warning "PROXY URL scheme is not http/https; the script will ignore it. Use an HTTP(S) proxy or run an HTTP-to-SOCKS relay."
        }
    }

    Write-Host "Installing Python dependencies from requirements.txt..."
    python -m pip install -r requirements.txt

    Write-Host "Done. To activate the venv in this or future sessions run: .\.venv\Scripts\Activate.ps1"
    Write-Host "If installation fails on compiling wheels, install Visual C++ Build Tools or try Python 3.10/3.11 which have wider binary wheel support."
    exit 0
}
catch {
    Write-Error "Installation failed: $_"
    exit 2
}
