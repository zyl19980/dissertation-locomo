#!/bin/bash
# Evaluate Ollama Local Models
# This script evaluates Ollama models on the LoCoMo QA task

# sets necessary environment variables
source scripts/env.sh

# Model configuration
# You can change the model name to any Ollama model you have installed
# Examples: qwen3:8b, qwen2.5:7b, qwen2.5:3b, llama3, mistral, etc.
OLLAMA_MODEL="qwen3:8b"

# Batch size (recommended: 1 for better stability, can try higher values for speed)
BATCH_SIZE=1

echo "========================================"
echo "Evaluating Ollama Model: $OLLAMA_MODEL"
echo "========================================"
echo "Data file: $DATA_FILE_PATH"
echo "Output file: $OUT_DIR/ollama_${OLLAMA_MODEL//[:.]/_}_${QA_OUTPUT_FILE}"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "Make sure Ollama service is running!"
echo "You can check with: curl http://localhost:11434/api/tags"
echo "========================================"
echo ""

# Run evaluation
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH \
    --out-file $OUT_DIR/ollama_${OLLAMA_MODEL//[:.]/_}_${QA_OUTPUT_FILE} \
    --model $OLLAMA_MODEL \
    --batch-size $BATCH_SIZE

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: $OUT_DIR/ollama_${OLLAMA_MODEL//[:.]/_}_${QA_OUTPUT_FILE}"
echo "Statistics saved to: $OUT_DIR/ollama_${OLLAMA_MODEL//[:.]/_}_${QA_OUTPUT_FILE%.json}_stats.json"
echo "========================================"
