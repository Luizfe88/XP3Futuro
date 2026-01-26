@echo off
echo.
echo ==================================================
echo     XP3 PRO - RESET MACHINE LEARNING (LIMPEZA)
echo ==================================================
echo.
echo ATENCAO: Isso vai zerar o aprendizado do Q-Learning
echo          para comecar do zero na conta real.
echo.
pause

:: Define data atual para nome da pasta de backup
set "DATA=%date:~6,4%-%date:~3,2%-%date:~0,2%"
set "BACKUP_PASTA=backup_ml_%DATA%"

:: Cria pasta de backup
if not exist "%BACKUP_PASTA%" mkdir "%BACKUP_PASTA%"

echo.
echo Criando backup em: %BACKUP_PASTA%
echo.

:: Lista de arquivos a fazer backup e deletar
set "ARQUIVOS=qtable.npy ml_trade_history.json symbol_loss_streak.json adaptive_weights.json"

for %%f in (%ARQUIVOS%) do (
    if exist "%%f" (
        echo Backing up: %%f
        move "%%f" "%BACKUP_PASTA%\%%f" >nul
        echo    -> Movido para backup
    ) else (
        echo %%f nao existe (pulando)
    )
)

echo.
echo ==================================================
echo          RESET CONCLUIDO COM SUCESSO!
echo ==================================================
echo.
echo Arquivos zerados:
echo - qtable.npy
echo - ml_trade_history.json
echo - symbol_loss_streak.json (opcional)
echo - adaptive_weights.json (opcional)
echo.
echo Backup salvo em: %BACKUP_PASTA%
echo.
echo Agora voce pode ligar o bot na conta real com ML limpo!
echo.
pause