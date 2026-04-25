param(
    [switch]$GenerateSampleIfMissing
)

$command = "python -m src.risk_model.pipeline.weekly_job"
if ($GenerateSampleIfMissing) {
    $command += " --generate-sample-if-missing"
}

Write-Output "Executing: $command"
Invoke-Expression $command
