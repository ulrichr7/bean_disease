# Run Locust headless and save CSV outputs
# Usage: .\run_locust.ps1 -users 100 -spawnRate 10 -runTime "1m"
param(
    [int]$users = 100,
    [int]$spawnRate = 10,
    [string]$runTime = "1m",
    [string]$host = "http://localhost:8001"
)

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$csvPrefix = "locust-results-$timestamp"

locust -f locustfile.py --headless --users $users --spawn-rate $spawnRate --run-time $runTime --host $host --csv $csvPrefix

Write-Host "Locust run complete. CSV files prefixed with: $csvPrefix"
