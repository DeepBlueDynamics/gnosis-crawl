<#
.SYNOPSIS
    Launch or teardown self-hosted Firecrawl for combat benchmarks.

.EXAMPLE
    .\combat\firecrawl.ps1 up      # pull images, start, wait for healthy
    .\combat\firecrawl.ps1 down    # stop and remove containers + volumes
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("up", "down")]
    [string]$Action = "up"
)

$ErrorActionPreference = "Stop"
$ComposeFile = Join-Path $PSScriptRoot "firecrawl-compose.yaml"
$FcUrl = "http://localhost:3002"

function Start-Firecrawl {
    if (-not (Test-Path $ComposeFile)) {
        Write-Host "[firecrawl] Missing $ComposeFile" -ForegroundColor Red
        exit 1
    }

    Write-Host "[firecrawl] Starting docker compose..." -ForegroundColor Cyan
    docker compose -f $ComposeFile up -d --pull always

    # Wait for healthy
    Write-Host "[firecrawl] Waiting for API at $FcUrl ..." -ForegroundColor Cyan
    $retries = 40
    for ($i = 0; $i -lt $retries; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $FcUrl -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($resp.StatusCode -lt 500) {
                Write-Host "[firecrawl] Ready!" -ForegroundColor Green
                Write-Host ""
                Write-Host "  API key:    fc-combat-test" -ForegroundColor Yellow
                Write-Host "  Endpoint:   $FcUrl" -ForegroundColor Yellow
                Write-Host "  Run arena:  pytest combat/ -m combat -v -s" -ForegroundColor Yellow
                return
            }
        }
        catch {}
        Start-Sleep -Seconds 3
    }
    Write-Host "[firecrawl] Timed out waiting for API" -ForegroundColor Red
    docker compose -f $ComposeFile logs api --tail 30
    exit 1
}

function Stop-Firecrawl {
    if (-not (Test-Path $ComposeFile)) {
        Write-Host "[firecrawl] Missing $ComposeFile" -ForegroundColor Yellow
        return
    }

    Write-Host "[firecrawl] Stopping containers..." -ForegroundColor Cyan
    docker compose -f $ComposeFile down -v

    Write-Host "[firecrawl] Cleaned up." -ForegroundColor Green
}

switch ($Action) {
    "up"   { Start-Firecrawl }
    "down" { Stop-Firecrawl }
}
