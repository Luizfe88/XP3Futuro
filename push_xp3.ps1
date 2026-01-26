param(
  [string]$Message = $("auto: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
)

function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host $msg -ForegroundColor Green }
function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host $msg -ForegroundColor Red }

try {
  if ($PSScriptRoot) {
    Set-Location $PSScriptRoot
  } else {
    Set-Location "C:\Users\luizf\Documents\xp3v5"
  }
} catch {
  Write-Err "Falha ao acessar o diretório do projeto: $_"
  exit 1
}

function Ensure-CredentialHelper {
  try {
    $helper = git config --global credential.helper
    if ([string]::IsNullOrWhiteSpace($helper)) {
      Write-Info "Configurando Git Credential Manager (Windows)..."
      git config --global credential.helper manager-core | Out-Null
    }
  } catch {
    Write-Warn "Não foi possível configurar o credential.helper automaticamente."
  }
}

function Ensure-Remote {
  param([string]$expectedUrl = "https://github.com/Luizfe88/xp3v5.git")
  try {
    $origin = git remote get-url origin 2>$null
    if ([string]::IsNullOrWhiteSpace($origin)) {
      Write-Info "Configurando remote 'origin' para $expectedUrl..."
      git remote add origin $expectedUrl | Out-Null
    } elseif ($origin -ne $expectedUrl) {
      Write-Warn "Remote origin atual: $origin"
      Write-Info "Atualizando para $expectedUrl..."
      git remote set-url origin $expectedUrl | Out-Null
    }
  } catch {
    Write-Err "Falha ao garantir remote origin: $_"
    exit 1
  }
}

function Get-CurrentBranch {
  try {
    $branch = git rev-parse --abbrev-ref HEAD 2>$null
    if ([string]::IsNullOrWhiteSpace($branch) -or $branch -eq "HEAD") {
      # Fallback para main
      return "main"
    }
    return $branch.Trim()
  } catch {
    return "main"
  }
}

Write-Info "Inicializando push automático (xp3v5)..."
Ensure-CredentialHelper
Ensure-Remote

# Garante branch atual
$branch = Get-CurrentBranch
Write-Info "Branch detectado: $branch"

# Previne abortar em CRLF/auto
git config core.autocrlf true | Out-Null

# Stage mudanças
Write-Info "Adicionando arquivos..."
git add -A

# Commit (permite vazio para forçar push)
Write-Info "Criando commit: $Message"
git commit --allow-empty -m $Message

# Push
Write-Info "Enviando para origin/$branch..."
try {
  git push origin $branch
  if ($LASTEXITCODE -eq 0) {
    Write-Ok "Push concluído com sucesso para origin/$branch."
    exit 0
  } else {
    throw "git push retornou código $LASTEXITCODE"
  }
} catch {
  Write-Err "Falha no push: $_"
  Write-Warn "Se ver 403 (Permission Denied), verifique credenciais do GitHub:"
  Write-Host "1) Instale/atualize Git Credential Manager (já padrão no Git para Windows)"
  Write-Host "2) Execute: git config --global credential.helper manager-core"
  Write-Host "3) Na próxima tentativa de push, informe usuário 'Luizfe88' e use um PAT como senha"
  Write-Host "   Crie um PAT em github.com > Settings > Developer settings > Personal access tokens (classic)"
  Write-Host "   Escopos mínimos: repo"
  exit 1
}
