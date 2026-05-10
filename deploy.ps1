#!/usr/bin/env pwsh
<#
.SYNOPSIS
Deployment script for grub-crawl service
Based on gnosis-ocr deploy.ps1 pattern

.DESCRIPTION
Deploys grub-crawl to local Docker, Cloud Run, or a 2-node mesh.
Supports local development, cloud production, and mesh topologies.

.PARAMETER Target
Deployment target: 'local', 'cloudrun', or 'mesh'

.PARAMETER Tag
Docker image tag (default: 'latest')

.PARAMETER Rebuild
Force rebuild of Docker image

.PARAMETER MeshSecret
Shared HMAC secret for mesh nodes (default: auto-generated)

.PARAMETER CloudMeshPeer
URL of a cloud peer to connect to when deploying with mesh enabled

.EXAMPLE
./deploy.ps1 -Target local
Deploy single node to local Docker Compose

.EXAMPLE
./deploy.ps1 -Target mesh
Deploy 2-node mesh locally (node-a + node-b)

.EXAMPLE
./deploy.ps1 -Target cloudrun -Tag v1.0.0
Deploy to Google Cloud Run

.EXAMPLE
./deploy.ps1 -Target cloudrun -Tag v1.0.0 -CloudMeshPeer http://localhost:6792 -MeshSecret mykey
Deploy to Cloud Run with mesh enabled, peering with local node
#>

param(
    [ValidateSet("local", "cloudrun", "mesh", "dockerhub")]
    [string]$Target = "local",

    [string]$Tag = "latest",

    [switch]$Rebuild = $false,

    [string]$MeshSecret = "",

    [string]$CloudMeshPeer = "",

    [string]$DockerHubRepo = "deepbluedynamics/grubcrawler"
)

# Configuration — matches grub-site / gnosis-ocr pattern
$ServiceName = "grub-crawl"
$ProjectId = $env:GOOGLE_CLOUD_PROJECT
if (-not $ProjectId) { $ProjectId = "gnosis-459403" }
$Region = "us-central1"
$ImageName = "gcr.io/${ProjectId}/${ServiceName}"

Write-Host "==> Deploying $ServiceName to $Target" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if ($Target -eq "cloudrun") {
    # Check gcloud
    try {
        $null = gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Please authenticate with: gcloud auth login"
            exit 1
        }
    }
    catch {
        Write-Error "gcloud CLI not found. Install: winget install Google.CloudSDK"
        exit 1
    }
    gcloud config set project $ProjectId
}

# ---------------------------------------------------------------------------
# Build image
# ---------------------------------------------------------------------------
Write-Host "==> Building Docker image..." -ForegroundColor Yellow

if ($Target -eq "cloudrun") {
    $FullImageName = "${ImageName}:${Tag}"
} elseif ($Target -eq "dockerhub") {
    $FullImageName = "${DockerHubRepo}:${Tag}"
} else {
    $FullImageName = "${ServiceName}:${Tag}"
}

$BuildArgs = @()
if ($Rebuild) { $BuildArgs += "--no-cache" }

try {
    docker build $BuildArgs -t $FullImageName .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed"
        exit 1
    }
    Write-Host "==> Image built: $FullImageName" -ForegroundColor Green
}
catch {
    Write-Error "Failed to build Docker image: $_"
    exit 1
}

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------
switch ($Target) {
    "local" {
        Write-Host "==> Deploying single node locally..." -ForegroundColor Yellow

        # Stop mesh compose first (shares port 6792)
        docker-compose -f docker-compose.mesh.yml down 2>$null
        docker-compose down 2>$null

        try {
            docker-compose up -d
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Docker Compose failed"
                exit 1
            }

            Write-Host ""
            Write-Host "==> Service deployed locally" -ForegroundColor Green
            Write-Host "    API:      http://localhost:6792" -ForegroundColor Cyan
            Write-Host "    Health:   http://localhost:6792/health" -ForegroundColor Cyan
            Write-Host "    Site:     http://localhost:6792/site" -ForegroundColor Cyan
            Write-Host ""
            docker-compose logs --tail=10
        }
        catch {
            Write-Error "Failed to deploy locally: $_"
            exit 1
        }
    }

    "mesh" {
        Write-Host "==> Deploying 2-node mesh locally..." -ForegroundColor Yellow

        # Stop single-node compose first (shares port 6792)
        docker-compose down 2>$null
        docker-compose -f docker-compose.mesh.yml down 2>$null

        try {
            docker-compose -f docker-compose.mesh.yml up -d --build
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Mesh compose failed"
                exit 1
            }

            Write-Host ""
            Write-Host "==> Mesh deployed (2 nodes)" -ForegroundColor Green
            Write-Host "    Node A:   http://localhost:6792  (local)" -ForegroundColor Cyan
            Write-Host "    Node B:   http://localhost:6793  (cloud)" -ForegroundColor Cyan
            Write-Host "    Peers A:  http://localhost:6792/mesh/peers" -ForegroundColor Cyan
            Write-Host "    Peers B:  http://localhost:6793/mesh/peers" -ForegroundColor Cyan
            Write-Host "    Health:   http://localhost:6792/health" -ForegroundColor Cyan
            Write-Host "    Site:     http://localhost:6792/site" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "    Verify:   curl http://localhost:6792/mesh/peers" -ForegroundColor Yellow
            Write-Host ""

            # Wait for startup then show peer status
            Start-Sleep -Seconds 5
            docker-compose -f docker-compose.mesh.yml logs --tail=15
        }
        catch {
            Write-Error "Failed to deploy mesh: $_"
            exit 1
        }
    }

    "cloudrun" {
        Write-Host "==> Deploying to Google Cloud Run..." -ForegroundColor Yellow

        # Enable required APIs (one-time)
        Write-Host "==> Enabling required APIs..." -ForegroundColor Yellow
        gcloud services enable `
            run.googleapis.com `
            cloudbuild.googleapis.com `
            artifactregistry.googleapis.com

        # Push image
        Write-Host "==> Pushing image to Container Registry..." -ForegroundColor Yellow
        try {
            docker push $FullImageName
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to push image"
                exit 1
            }
        }
        catch {
            Write-Error "Failed to push Docker image: $_"
            exit 1
        }

        # Build env vars
        $EnvVars = @(
            "RUNNING_IN_CLOUD=true",
            "GCS_BUCKET_NAME=grub-crawl-storage-prod",
            "GNOSIS_AUTH_URL=https://auth.nuts.services",
            "GOOGLE_CLOUD_PROJECT=$ProjectId"
        )

        # Add mesh env vars if mesh peer specified
        if ($CloudMeshPeer) {
            if (-not $MeshSecret) {
                $MeshSecret = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ })
                Write-Host "==> Generated mesh secret: $MeshSecret" -ForegroundColor Yellow
                Write-Host "    Use this secret when starting the local peer." -ForegroundColor Yellow
            }

            # Get the Cloud Run service URL (will be set after deploy, use placeholder)
            $EnvVars += "MESH_ENABLED=true"
            $EnvVars += "MESH_NODE_NAME=cloud"
            $EnvVars += "MESH_SECRET=$MeshSecret"
            $EnvVars += "MESH_PEERS=$CloudMeshPeer"
            $EnvVars += "MESH_PREFER_LOCAL=false"
            $EnvVars += "MESH_HEARTBEAT_INTERVAL_S=15"
        }

        # Deploy to Cloud Run
        Write-Host "==> Deploying to Cloud Run..." -ForegroundColor Yellow

        $EnvString = $EnvVars -join ","

        try {
            gcloud run deploy $ServiceName `
                --image $FullImageName `
                --platform managed `
                --region $Region `
                --allow-unauthenticated `
                --port 8080 `
                --memory 1Gi `
                --cpu 1 `
                --max-instances 10 `
                --timeout 300 `
                --concurrency 100 `
                --set-env-vars $EnvString

            if ($LASTEXITCODE -ne 0) {
                Write-Error "Cloud Run deployment failed"
                exit 1
            }

            # Get service URL
            $ServiceUrl = gcloud run services describe $ServiceName --region $Region --format "value(status.url)"

            # If mesh is enabled, update MESH_ADVERTISE_URL to the actual Cloud Run URL
            if ($CloudMeshPeer) {
                Write-Host "==> Updating mesh advertise URL..." -ForegroundColor Yellow
                gcloud run services update $ServiceName `
                    --region $Region `
                    --update-env-vars "MESH_ADVERTISE_URL=$ServiceUrl"
            }

            Write-Host ""
            Write-Host "==> Service deployed to Cloud Run" -ForegroundColor Green
            Write-Host "    URL:      $ServiceUrl" -ForegroundColor Cyan
            Write-Host "    Health:   $ServiceUrl/health" -ForegroundColor Cyan
            Write-Host "    Site:     $ServiceUrl/site" -ForegroundColor Cyan

            if ($CloudMeshPeer) {
                Write-Host ""
                Write-Host "==> Mesh enabled" -ForegroundColor Green
                Write-Host "    Peers:    $ServiceUrl/mesh/peers" -ForegroundColor Cyan
                Write-Host "    Status:   $ServiceUrl/mesh/status" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "    To connect your local node:" -ForegroundColor Yellow
                Write-Host "    MESH_ENABLED=true MESH_SECRET=$MeshSecret MESH_PEERS=$ServiceUrl \" -ForegroundColor Yellow
                Write-Host "      MESH_ADVERTISE_URL=http://your-local-ip:6792 \" -ForegroundColor Yellow
                Write-Host "      uvicorn app.main:app --port 6792" -ForegroundColor Yellow
            }
        }
        catch {
            Write-Error "Failed to deploy to Cloud Run: $_"
            exit 1
        }
    }

    "dockerhub" {
        Write-Host "==> Pushing image to Docker Hub: $FullImageName" -ForegroundColor Yellow

        # Verify Docker Hub login
        $dockerInfo = docker info 2>$null | Out-String
        if ($dockerInfo -notmatch "Username:") {
            Write-Host "==> Not logged into Docker Hub — running 'docker login'..." -ForegroundColor Yellow
            docker login
            if ($LASTEXITCODE -ne 0) {
                Write-Error "docker login failed"
                exit 1
            }
        } else {
            $username = ($dockerInfo | Select-String -Pattern "Username:\s*(.+)").Matches.Groups[1].Value.Trim()
            Write-Host "==> Logged in as: $username" -ForegroundColor Green
        }

        # Also tag with a date stamp so old releases stay reachable
        $DateTag = Get-Date -Format "yyyyMMdd-HHmm"
        $DatedImage = "${DockerHubRepo}:${DateTag}"

        try {
            docker tag $FullImageName $DatedImage
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to tag image with date stamp"
                exit 1
            }
            Write-Host "==> Tagged: $DatedImage" -ForegroundColor Green

            docker push $FullImageName
            if ($LASTEXITCODE -ne 0) {
                Write-Error "docker push $FullImageName failed"
                exit 1
            }

            docker push $DatedImage
            if ($LASTEXITCODE -ne 0) {
                Write-Error "docker push $DatedImage failed"
                exit 1
            }

            Write-Host ""
            Write-Host "==> Pushed to Docker Hub" -ForegroundColor Green
            Write-Host "    docker pull $FullImageName" -ForegroundColor Cyan
            Write-Host "    docker pull $DatedImage" -ForegroundColor Cyan
            Write-Host "    https://hub.docker.com/r/${DockerHubRepo}" -ForegroundColor Cyan
        }
        catch {
            Write-Error "Failed to push to Docker Hub: $_"
            exit 1
        }
    }
}

Write-Host ""
Write-Host "==> Deployment completed." -ForegroundColor Green
