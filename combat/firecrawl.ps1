<#
.SYNOPSIS
    Launch or teardown self-hosted Firecrawl for combat benchmarks.

.EXAMPLE
    .\combat\firecrawl.ps1 up      # clone, start, wait for healthy
    .\combat\firecrawl.ps1 down    # stop and remove containers + images
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("up", "down")]
    [string]$Action = "up"
)

$ErrorActionPreference = "Stop"
$FcDir = Join-Path $PSScriptRoot ".firecrawl"
$FcUrl = "http://localhost:3002"

function Start-Firecrawl {
    # Clone if needed
    if (-not (Test-Path $FcDir)) {
        Write-Host "[firecrawl] Cloning repo..." -ForegroundColor Cyan
        git clone --depth 1 https://github.com/mendableai/firecrawl.git $FcDir
    }

    # Firecrawl needs a TEST_API_KEY in .env
    $envFile = Join-Path $FcDir ".env"
    if (-not (Test-Path $envFile)) {
        Write-Host "[firecrawl] Creating .env with test API key..." -ForegroundColor Cyan
        @"
NUM_WORKERS_PER_QUEUE=1
PORT=3002
HOST=0.0.0.0
TEST_API_KEY=fc-combat-test
BULL_AUTH_KEY=combat
"@ | Set-Content $envFile
    }

    # Start the stack
    Write-Host "[firecrawl] Starting docker compose..." -ForegroundColor Cyan
    Push-Location $FcDir
    try {
        docker compose up -d
    }
    finally {
        Pop-Location
    }

    # Wait for healthy
    Write-Host "[firecrawl] Waiting for API at $FcUrl ..." -ForegroundColor Cyan
    $retries = 30
    for ($i = 0; $i -lt $retries; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $FcUrl -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($resp.StatusCode -lt 500) {
                Write-Host "[firecrawl] Ready!" -ForegroundColor Green
                Write-Host ""
                Write-Host "  Set env for combat:  `$env:FIRECRAWL_URL = '$FcUrl'" -ForegroundColor Yellow
                Write-Host "  Run arena:           pytest combat/ -m combat -v -s" -ForegroundColor Yellow
                return
            }
        }
        catch {}
        Start-Sleep -Seconds 2
    }
    Write-Host "[firecrawl] Timed out waiting for API" -ForegroundColor Red
    exit 1
}

function Stop-Firecrawl {
    if (-not (Test-Path $FcDir)) {
        Write-Host "[firecrawl] Nothing to tear down (no .firecrawl dir)" -ForegroundColor Yellow
        return
    }

    Write-Host "[firecrawl] Stopping containers..." -ForegroundColor Cyan
    Push-Location $FcDir
    try {
        docker compose down -v --rmi local
    }
    finally {
        Pop-Location
    }

    Write-Host "[firecrawl] Removing clone..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force $FcDir

    Write-Host "[firecrawl] Cleaned up." -ForegroundColor Green
}

switch ($Action) {
    "up"   { Start-Firecrawl }
    "down" { Stop-Firecrawl }
}
