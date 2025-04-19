param (
    [string]$inputFile,
    [string]$workingDir,
    [int]$timeout = 6
)

# Change to the specified working directory
Set-Location -Path $workingDir

# Define the path to the xfoil executable based on the script's location
$scriptDir = $PSScriptRoot
$command = Join-Path $scriptDir "xfoil.exe"

# Resolve the input file path to an absolute path
$inputFilePath = Resolve-Path -Path (Join-Path $scriptDir $inputFile)

# Check if the input file exists
if (-not $inputFile -or -not (Test-Path $inputFilePath)) {
    Write-Error "Input file is required and must exist."
    exit 1
}

# Run the command in the background
$job = Start-Job -ScriptBlock {
    param ($command, $inputFilePath, $workingDir)
    try {
        # Change to the working directory within the job
        Set-Location -Path $workingDir

        # Use Get-Content to read the input file and pipe it to the command
        Get-Content $inputFilePath | & $command
        if ($LASTEXITCODE -eq 0) {
            Write-Output "Command completed successfully."
        } else {
            Write-Output "Command failed with exit code $LASTEXITCODE."
        }
    } catch {
        Write-Output "Command execution failed: $_"
    }
} -ArgumentList $command, $inputFilePath, $workingDir -Name "WatchdogJob"

# Function to kill the xfoil process if it exceeds the timeout
function Watchdog {
    param ($timeout, $jobName)
    Start-Sleep -Seconds $timeout
    $job = Get-Job -Name $jobName
    if ($job.State -eq "Running") {
        Write-Output "Command exceeded timeout. Stopping job."
        
        # Stop the job first
        Stop-Job -Name $jobName -Force

        # Ensure the xfoil.exe process is terminated
        $xfoilProcess = Get-Process -Name "xfoil" -ErrorAction SilentlyContinue
        if ($xfoilProcess) {
            $xfoilProcess | ForEach-Object {
                Stop-Process -Id $_.Id -Force
                Write-Output "Terminated xfoil.exe process with ID $($_.Id)"
            }
        } else {
            Write-Output "No xfoil.exe process found to terminate."
        }
    }
}

# Start the watchdog timer
$watchdogJob = Start-Job -ScriptBlock { Watchdog $using:timeout "WatchdogJob" }

# Wait for the command to complete
Wait-Job -Name "WatchdogJob" -Timeout $timeout

# Check the exit status of the command
$job = Get-Job -Name "WatchdogJob"
if ($job.State -eq "Completed") {
    $output = Receive-Job -Name "WatchdogJob"
    Write-Output $output
} else {
    Write-Output "Command failed or was stopped."
}

# Clean up
Remove-Job -Job $job -Force
Remove-Job -Job $watchdogJob -Force
