# 一键生成提交用实验报告：Markdown -> Word -> PDF，并同步到作业文件夹根目录
#
# 用法：.\build_report.ps1
#        .\build_report.ps1 -Detailed    # 生成 17 页详细版

param(
    [switch]$Detailed
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $project

$source = if ($Detailed) { "report/实验报告_详细版.md" } else { "report/实验报告.md" }
$target = if ($Detailed) { "report/实验报告_详细版.docx" } else { "report/实验报告.docx" }
$pdf = [System.IO.Path]::ChangeExtension((Resolve-Path -Relative $target), ".pdf")

Write-Host "[1/4] Markdown -> Word"
pandoc $source -f gfm -t docx -o $target --resource-path=report
if ($LASTEXITCODE -ne 0) { throw "pandoc 转换失败" }

Write-Host "[2/4] 排版"
$styleArgs = @()
if (-not $Detailed) { $styleArgs = @("--compact") }
python scripts/make_report_docx.py $target @styleArgs
if ($LASTEXITCODE -ne 0) { throw "排版失败" }

Write-Host "[3/4] 导出 PDF"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open((Resolve-Path $target).Path, $false, $true)
    $doc.SaveAs([ref](Join-Path $project $pdf), [ref]17)
    $pages = $doc.ComputeStatistics(2)
    $doc.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
Write-Host "      $pdf（$pages 页）"

if (-not $Detailed) {
    Write-Host "[4/4] 同步到作业文件夹根目录"
    Copy-Item (Join-Path $project $pdf) (Join-Path (Split-Path -Parent $project) "实验报告.pdf") -Force
    Write-Host "      $(Split-Path -Parent $project)\实验报告.pdf"
}

if ($pages -gt 5 -and -not $Detailed) {
    Write-Warning "报告超过 5 页（当前 $pages 页），会被扣分，请精简内容"
}
