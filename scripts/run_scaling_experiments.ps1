# scripts/run_scaling_experiments.ps1
# -----------------------------------------------------------------------
# Windows PowerShell scaling experiments sweep script for Parallel K-Means.
# Runs strong-scaling or weak-scaling sweeps and outputs CSV metrics.
#
# Usage:
#   .\scripts\run_scaling_experiments.ps1 -Mode strong -Dataset data/synth_10k.bin -K 5
#   .\scripts\run_scaling_experiments.ps1 -Mode weak -Dataset data/synth_base -K 5 -Ranks 1,2,4
# -----------------------------------------------------------------------
param(
    [string]$Mode = "strong",
    [string]$Dataset = "data/test_10k.bin",
    [int]$K = 5,
    [int]$MaxIters = 100,
    [double]$Tol = 1e-4,
    [int]$Seed = 42,
    [int[]]$Ranks = @(1, 2, 4),
    [string]$Bin = "bin/kmeans_mpi.exe",
    [string]$ResultsDir = "results"
)

if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$CsvPath = Join-Path $ResultsDir "scaling_${Mode}_${Timestamp}.csv"
"mode,ranks,dataset,N,D,K,iterations,SSE,total_time_s,compute_avg_s,comm_avg_s,comm_overhead_fraction" | Out-File -FilePath $CsvPath -Encoding utf8

foreach ($r in $Ranks) {
    $dsPath = $Dataset
    if ($Mode -eq "weak") {
        $dsPath = "${Dataset}_r${r}.bin"
    }

    $outPrefix = Join-Path $ResultsDir "${Mode}_r${r}"
    Write-Host ">>> [$Mode] ranks=$r dataset=$dsPath"

    $process = Start-Process -FilePath "mpirun" -ArgumentList "-np $r $Bin $dsPath $K $MaxIters $Tol $Seed $outPrefix" -NoNewWindow -Wait -PassThru

    $metricsPath = "${outPrefix}_metrics.txt"
    if (Test-Path $metricsPath) {
        $lines = Get-Content $metricsPath
        $metrics = @{}
        foreach ($line in $lines) {
            $parts = $line.Split(" ", 2)
            if ($parts.Count -eq 2) {
                $metrics[$parts[0].Trim()] = $parts[1].Trim()
            }
        }
        $row = "$Mode,$r,$dsPath,$($metrics['N']),$($metrics['D']),$K,$($metrics['iterations']),$($metrics['SSE']),$($metrics['total_time_s']),$($metrics['compute_time_avg_s']),$($metrics['comm_time_avg_s']),$($metrics['comm_overhead_fraction'])"
        $row | Out-File -FilePath $CsvPath -Append -Encoding utf8
    } else {
        Write-Warning "Failed to read metrics file $metricsPath"
    }
}

Write-Host "Results written to $CsvPath"
