# Lanceur local de l'interface : aucune modification de PATH ni de politique
# PowerShell. Le .bat voisin reste utilisable si les scripts .ps1 sont bloques.
$ErrorActionPreference = 'Stop'

function Find-ProjectPython {
    # On privilegie l'installation qui a deja fonctionne sur le poste Windows.
    # LOCALAPPDATA evite de coder un nom d'utilisateur dans le projet partage.
    $candidates = @()
    if ($env:LOCALAPPDATA) {
        $usualPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
        if (Test-Path -LiteralPath $usualPython -PathType Leaf) {
            $candidates += @{ Executable = $usualPython; Prefix = @() }
        }
    }
    foreach ($commandName in @('python', 'py')) {
        $command = Get-Command $commandName -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $command) {
            $prefixArguments = @()
            if ($commandName -eq 'py') { $prefixArguments = @('-3') }
            $candidates += @{ Executable = $command.Source; Prefix = $prefixArguments }
        }
    }

    foreach ($candidate in $candidates) {
        # Une commande existante peut etre un alias du Microsoft Store.
        # L'execution d'un vrai programme Python >= 3.10 sert de verification.
        try {
            $candidateArguments = $candidate.Prefix
            $result = & $candidate.Executable @candidateArguments -c "import sys; print(sys.executable); sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $result) {
                $resolvedPath = ([string]($result | Select-Object -Last 1)).Trim()
                if (Test-Path -LiteralPath $resolvedPath -PathType Leaf) {
                    return $resolvedPath
                }
            }
        }
        catch {
            # Un candidat inutilisable n'empeche pas d'essayer le suivant.
        }
    }
    throw 'Python >= 3.10 introuvable. Installer Python depuis python.org, puis relancer ce fichier.'
}

try {
    # Le lancement fonctionne aussi depuis un autre dossier que celui du projet.
    Set-Location -LiteralPath $PSScriptRoot
    $projectPython = Find-ProjectPython
    Write-Host "Python utilise : $projectPython"
    Write-Host 'Installation / verification des dependances...'
    & $projectPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    $installationExitCode = $LASTEXITCODE
    if ($installationExitCode -ne 0) {
        # Garder le message de pip ci-dessus et son code de sortie original.
        Write-Host "Installation interrompue (code $installationExitCode). Corriger le message de pip ci-dessus, puis relancer."
        exit $installationExitCode
    }

    Write-Host 'Ouverture de Streamlit. Garder ce terminal ouvert ; Ctrl+C pour arreter.'
    & $projectPython -m streamlit run (Join-Path $PSScriptRoot 'app.py')
    $applicationExitCode = $LASTEXITCODE
    if ($applicationExitCode -ne 0) {
        Write-Host "Streamlit s'est termine avec le code $applicationExitCode. Voir le message ci-dessus."
    }
    exit $applicationExitCode
}
catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
