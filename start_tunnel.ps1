param(
    [int]$Port = 8000,
    [string]$HostAddr = "0.0.0.0",
    [string]$NgrokRegion = "ap",
    [bool]$AutoReload = $false
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Stop-IfRunning {
    param($proc)
    if ($null -ne $proc) {
        try {
            if (-not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
        } catch {}
    }
}

$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

try {
    $ngrokCmd = (Get-Command ngrok -ErrorAction Stop).Source
} catch {
    $candidates = @(
        (Join-Path $root "ngrok.exe"),
        (Join-Path $env:USERPROFILE "ngrok.exe"),
        (Join-Path $env:USERPROFILE "Downloads\ngrok.exe"),
        (Join-Path $env:LOCALAPPDATA "ngrok\ngrok.exe")
    ) | Where-Object { $_ -and (Test-Path $_) }

    if (@($candidates).Count -gt 0) {
        $ngrokCmd = @($candidates)[0]
        Write-Host "[INFO] Using ngrok at: $ngrokCmd"
    } else {
        Write-Error "ngrok not found. Install ngrok and run 'ngrok config add-authtoken <token>' first."
        Write-Host "Tip: If already installed, add it to PATH or place ngrok.exe in this project folder."
        exit 1
    }
}

$apiProc = $null
$ngrokProc = $null
$restartCount = 0
$maxRestarts = 50
$supportsRegionFlag = $false
Write-Host "[INFO] Probing ngrok features..."
$probeOut = Join-Path $env:TEMP ("ngrok_help_{0}.out.txt" -f $PID)
$probeErr = Join-Path $env:TEMP ("ngrok_help_{0}.err.txt" -f $PID)
try {
    $probeProc = Start-Process -FilePath $ngrokCmd `
        -ArgumentList @("http", "--help") `
        -PassThru `
        -RedirectStandardOutput $probeOut `
        -RedirectStandardError $probeErr

    if (-not $probeProc.WaitForExit(4000)) {
        Stop-Process -Id $probeProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "[WARN] ngrok help probe timed out; continuing without --region flag."
    } else {
        $ngrokHelp = ""
        if (Test-Path $probeOut) { $ngrokHelp += (Get-Content -Path $probeOut -Raw) }
        if (Test-Path $probeErr) { $ngrokHelp += "`n" + (Get-Content -Path $probeErr -Raw) }
        $supportsRegionFlag = ($ngrokHelp -match "(?m)^\s*--region\b")
    }
} catch {
    Write-Host "[WARN] Could not probe ngrok help; continuing without --region flag."
} finally {
    Remove-Item -Path $probeOut -ErrorAction SilentlyContinue
    Remove-Item -Path $probeErr -ErrorAction SilentlyContinue
}

try {
    Write-Host "[START] Backend: http://$HostAddr`:$Port"
    $uvicornArgs = @("-m", "uvicorn", "api:app", "--host", $HostAddr, "--port", "$Port")
    if ($AutoReload) {
        # Limit noisy/hot paths that can trigger expensive reload scans or reload loops.
        $reloadExcludes = @(
            "weights/*",
            "qdrant_storage/*",
            "venv/*",
            ".git/*",
            "__pycache__/*",
            "*.onnx",
            "*.pt",
            "*.pth",
            "*.bin"
        )
        $uvicornArgs += @(
            "--reload",
            "--reload-dir", $root
        )
        foreach ($pattern in $reloadExcludes) {
            $uvicornArgs += @("--reload-exclude", $pattern)
        }
        Write-Host "[INFO] Auto-reload enabled."
    } else {
        Write-Host "[INFO] Auto-reload disabled for stable tunnel startup. Use -AutoReload `$true for development."
    }

    $apiProc = Start-Process -FilePath $python `
        -ArgumentList $uvicornArgs `
        -WorkingDirectory $root `
        -PassThru `
        -NoNewWindow

    $backendReady = $false
    $startupDeadline = (Get-Date).AddMinutes(3)
    $probeCount = 0
    Write-Host "[WAIT] Waiting for backend to start..."
    while ((Get-Date) -lt $startupDeadline) {
        $probeCount += 1
        $apiProc.Refresh()
        if ($apiProc.HasExited) {
            throw "Backend exited during startup. Check logs above for the error."
        }

        try {
            # Probe /status — root / returns 404 on FastAPI which throws on Invoke-WebRequest
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/status" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -lt 500) {
                $backendReady = $true
                break
            }
        } catch {
            # 404 on /status also means server is up — catch connection refused only
            $msg = $_.Exception.Message
            if ($msg -notmatch "refused|connect|timed out") {
                $backendReady = $true
                break
            }
        }

        Write-Host "[WAIT] ($probeCount) Backend starting, please wait..."
        Start-Sleep -Seconds 1
    }
    if (-not $backendReady) {
        Write-Host "[WARN] Backend did not respond in time; starting ngrok anyway."
    } else {
        Write-Host "[READY] Backend is reachable."
    }

    Write-Host "[START] Ngrok tunnel for port $Port"
    $ngrokArgs = @("http", "$Port")
    if ($supportsRegionFlag) {
        $ngrokArgs += @("--region", "$NgrokRegion")
        Write-Host "[INFO] Ngrok region: $NgrokRegion"
    } else {
        Write-Host "[INFO] Ngrok version does not support --region on 'http'; using default routing."
    }
    $ngrokProc = Start-Process -FilePath $ngrokCmd `
        -ArgumentList $ngrokArgs `
        -WorkingDirectory $root `
        -PassThru `
        -NoNewWindow

    Start-Sleep -Seconds 2
    try {
        $tunnels = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 3
        $public = $tunnels.tunnels | Where-Object { $_.public_url -like "https://*" } | Select-Object -First 1
        if ($public) {
            Write-Host "[NGROK] Public URL: $($public.public_url)"
            Write-Host "[NGROK] Open: $($public.public_url)/"
        }
    } catch {
        Write-Host "[NGROK] Tunnel started. Open ngrok web UI at http://127.0.0.1:4040"
    }

    Write-Host "[INFO] Press Ctrl+C to stop backend and ngrok."

    while ($true) {
        $apiProc.Refresh()
        $ngrokProc.Refresh()

        if ($ngrokProc.HasExited) {
            Write-Host "[STOP] Ngrok exited. Stopping backend..."
            break
        }

        if ($apiProc.HasExited) {
            if (-not $AutoReload) {
                Write-Host "[STOP] Backend exited. Stopping ngrok..."
                break
            }

            if ($restartCount -ge $maxRestarts) {
                Write-Host "[STOP] Backend restart limit reached ($maxRestarts). Stopping..."
                break
            }

            $restartCount += 1
            Write-Host "[WARN] Backend exited. Restarting ($restartCount/$maxRestarts)..."
            Start-Sleep -Seconds 1
            $apiProc = Start-Process -FilePath $python `
                -ArgumentList $uvicornArgs `
                -WorkingDirectory $root `
                -PassThru `
                -NoNewWindow
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Stop-IfRunning $ngrokProc
    Stop-IfRunning $apiProc
}