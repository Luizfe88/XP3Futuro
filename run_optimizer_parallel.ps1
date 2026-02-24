param(
  [string[]]$Symbols = @(),
  [int]$Days = 180,
  [int]$MaxEvals = 60,
  [int]$Bars = 5000
)

if ($Symbols.Count -eq 0) {
  $Symbols = @('WIN$N','IND$N','WDO$N','DOL$N','WSP$N','CCM$N','BGI$N','ICF$N','SFI$N','DI1$N','BIT$N','T10$N')
}

$ErrorActionPreference = "Stop"
$jobs = @()

foreach ($sym in $Symbols) {
  Write-Host ">>> 🔥 Disparando Worker para: $sym"
  $jobs += Start-Job -Name $sym -ScriptBlock {
    param($s,$days,$maxevals,$bars)
    $env:PYTHONUNBUFFERED = "1"
    py otimizador_semanal.py --mode run --days $days --maxevals $maxevals --bars $bars --symbols $s
  } -ArgumentList $sym,$Days,$MaxEvals,$Bars
}

Wait-Job -Job $jobs | Out-Null

foreach ($j in $jobs) {
  Write-Host ">>> 📥 Resultado do worker [$($j.Name)]"
  Receive-Job -Job $j
}

Remove-Job -Job $jobs -Force
