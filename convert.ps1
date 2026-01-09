Add-Content -Path "ml/src/data/XAU_1m_data_2026_additional.csv" -Value "Date;Open;High;Low;Close;Volume"

$lines = Get-Content "ml/src/data/XAUUSD_M1 (2).csv" | Select-Object -Skip 1

foreach ($line in $lines) {
    $parts = $line -split "`t"
    if ($parts.Count -ge 6) {
        try {
            $time = [DateTime]::ParseExact($parts[0], 'yyyy-MM-dd HH:mm:ss', $null)
            $dateStr = $time.ToString('yyyy.MM.dd HH:mm')
            $open = [double]$parts[1]
            $high = [double]$parts[2]
            $low = [double]$parts[3]
            $close = [double]$parts[4]
            $volume = [int][double]$parts[5]
            
            $outLine = "$dateStr;$($open.ToString('F6'));$($high.ToString('F6'));$($low.ToString('F6'));$($close.ToString('F6'));$volume"
            Add-Content -Path "ml/src/data/XAU_1m_data_2026_additional.csv" -Value $outLine
        }
        catch {
            Write-Host "Error: $_"
        }
    }
}
