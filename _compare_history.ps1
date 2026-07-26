$paths = @{
  "Hist_e30_full"        = "outputs\content_delta_pop5\ablation_itemmacro_3seed\full"
  "Hist_e30_wo_delta"    = "outputs\content_delta_pop5\ablation_itemmacro_3seed\wo_delta"
  "Hist_e30_wo_course"   = "outputs\content_delta_pop5\ablation_itemmacro_3seed\wo_course_feedback"
  "Hist_e30_wo_prereq"   = "outputs\content_delta_pop5\ablation_itemmacro_3seed\wo_prereq_aux"
  "Hist_e60_pseudo"      = "outputs\content_delta_pop5\pseudo_cold_itemmacro_v1\old_main_plus_pseudo_e60"
  "Tonight_runA_legacy"  = "outputs\content_delta_pop5\cold_patch_2026_05_19_20260519_0149\runA_legacy_baseline"
  "Tonight_runB_aux"     = "outputs\content_delta_pop5\cold_patch_2026_05_19_20260519_0149\runB_aux_hot_only"
  "Tonight_runC_aux_geo" = "outputs\content_delta_pop5\cold_patch_2026_05_19_20260519_0149\runC_aux_hot_only_geo"
}
$rows = @()
foreach ($name in $paths.Keys) {
  foreach ($s in 2025, 2026, 2027) {
    $csv = "$($paths[$name])\strict_item_cold_balanced_thr1_seed_$s\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    if (Test-Path $csv) {
      $r = Import-Csv $csv | Select-Object -First 1
      $rows += [PSCustomObject]@{
        Run      = $name
        Seed     = $s
        Cold_R10 = [math]::Round([double]$r.full_cold_item_macro_r10, 4)
        Cold_N10 = [math]::Round([double]$r.full_cold_item_macro_n10, 4)
        Hot_R10  = [math]::Round([double]$r.full_hot_item_macro_r10, 4)
        Hot_N10  = [math]::Round([double]$r.full_hot_item_macro_n10, 4)
      }
    }
  }
}

# DropoutNet reference
foreach ($s in 2025, 2026, 2027) {
  $drop = "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_$s\main_table_balanced_itemmacro_selector_v1\drop_static_result.json"
  if (Test-Path $drop) {
    $j = Get-Content $drop -Raw | ConvertFrom-Json
    $rows += [PSCustomObject]@{
      Run      = "DropoutNet"
      Seed     = $s
      Cold_R10 = [math]::Round([double]$j[0].full_cold_item_macro.'R@10', 4)
      Cold_N10 = [math]::Round([double]$j[0].full_cold_item_macro.'N@10', 4)
      Hot_R10  = [math]::Round([double]$j[0].full_hot_item_macro.'R@10', 4)
      Hot_N10  = [math]::Round([double]$j[0].full_hot_item_macro.'N@10', 4)
    }
  }
}

Write-Host "`n--- Per seed ---" -ForegroundColor Cyan
$rows | Sort-Object Run, Seed | Format-Table -AutoSize

Write-Host "`n--- Mean per Run ---" -ForegroundColor Cyan
$summary = $rows | Group-Object Run | ForEach-Object {
  [PSCustomObject]@{
    Run      = $_.Name
    nSeeds   = $_.Count
    Cold_R10 = [math]::Round(($_.Group | Measure-Object Cold_R10 -Average).Average, 4)
    Cold_N10 = [math]::Round(($_.Group | Measure-Object Cold_N10 -Average).Average, 4)
    Hot_R10  = [math]::Round(($_.Group | Measure-Object Hot_R10  -Average).Average, 4)
    Hot_N10  = [math]::Round(($_.Group | Measure-Object Hot_N10  -Average).Average, 4)
  }
}
$summary | Sort-Object Cold_N10 -Descending | Format-Table -AutoSize
