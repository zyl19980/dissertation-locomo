@echo off
REM Evaluate Ollama Local Models on Windows
REM This script evaluates Ollama models on the LoCoMo QA task

echo ========================================
echo Evaluating Ollama Model
echo ========================================

REM Configuration
set OUT_DIR=outputs
set DATA_FILE_PATH=data/locomo10.json
set QA_OUTPUT_FILE=locomo10_qa.json

REM Model configuration
REM You can change the model name to any Ollama model you have installed
REM Examples: qwen3:8b, qwen2.5:7b, qwen2.5:3b, llama3, mistral, etc.
set OLLAMA_MODEL=qwen3:8b

REM Batch size (recommended: 1 for better stability)
set BATCH_SIZE=1

REM Create output directory if it doesn't exist
if not exist %OUT_DIR% mkdir %OUT_DIR%

echo.
echo Configuration:
echo   Model: %OLLAMA_MODEL%
echo   Data file: %DATA_FILE_PATH%
echo   Output dir: %OUT_DIR%
echo   Batch size: %BATCH_SIZE%
echo.
echo Make sure Ollama service is running!
echo You can check with: curl http://localhost:11434/api/tags
echo ========================================
echo.

REM Replace special characters in model name for filename
set SAFE_MODEL_NAME=%OLLAMA_MODEL::=_%
set SAFE_MODEL_NAME=%SAFE_MODEL_NAME:.=_%

REM Run evaluation
python task_eval/evaluate_qa.py ^
    --data-file %DATA_FILE_PATH% ^
    --out-file %OUT_DIR%/ollama_%SAFE_MODEL_NAME%_%QA_OUTPUT_FILE% ^
    --model %OLLAMA_MODEL% ^
    --batch-size %BATCH_SIZE%

echo.
echo ========================================
echo Evaluation complete!
echo Results saved to: %OUT_DIR%/ollama_%SAFE_MODEL_NAME%_%QA_OUTPUT_FILE%
echo ========================================
echo.

pause
