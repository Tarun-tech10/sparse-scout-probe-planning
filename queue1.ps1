$ErrorActionPreference = "Continue"
$d = "c:\Users\tarun\OneDrive\Desktop\ERIS\sparse_scout_sol"
Set-Location $d
$log = "$d\logs"
if (-not (Test-Path $log)) { New-Item -ItemType Directory -Force $log | Out-Null }

$runs = @(
  @{tag="v2_e60_s192";  a=@("--epochs","60","--size","192")},
  @{tag="v2_e120_s192"; a=@("--epochs","120","--size","192")},
  @{tag="v2_e60_s256";  a=@("--epochs","60","--size","256","--bs","24")},
  @{tag="v2_e60_r34";   a=@("--epochs","60","--size","192","--arch","resnet34")}
)
foreach ($r in $runs) {
  $t = $r.tag
  Write-Output "=== START $t $(Get-Date -Format 'HH:mm:ss') ==="
  & python -u "$d\train2.py" --only-folds 1 --tag $t @($r.a) *> "$log\$t.log"
  Write-Output "=== DONE $t $(Get-Date -Format 'HH:mm:ss') ==="
  Get-Content "$log\$t.log" -Tail 14
}
Write-Output "ALL DONE"
