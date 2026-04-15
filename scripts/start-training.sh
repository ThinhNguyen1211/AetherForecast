#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-southeast-1}"
STACK_NAME="${STACK_NAME:-AetherForecastStack}"
TRAINING_STACK_NAME="${TRAINING_STACK_NAME:-$STACK_NAME}"
KEEP_INSTANCE_RUNNING="${KEEP_INSTANCE_RUNNING:-false}"
SSM_WAIT_ATTEMPTS="${SSM_WAIT_ATTEMPTS:-60}"
SSM_WAIT_INTERVAL_SECONDS="${SSM_WAIT_INTERVAL_SECONDS:-10}"
COMMAND_WAIT_INTERVAL_SECONDS="${COMMAND_WAIT_INTERVAL_SECONDS:-20}"

SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT}"
TIMEFRAME="${TIMEFRAME:-1h}"
TRAINING_HORIZON="${TRAINING_HORIZON:-7}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-96}"
MAX_ROWS_PER_SYMBOL="${MAX_ROWS_PER_SYMBOL:-20000}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BASE_MODEL_ID="${BASE_MODEL_ID:-amazon/chronos-2}"
BASE_MODEL_FALLBACK_ID="${BASE_MODEL_FALLBACK_ID:-amazon/chronos-t5-large}"

get_output() {
  local stack_name="$1"
  local key="$2"
  aws cloudformation describe-stacks \
    --stack-name "$stack_name" \
    --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='${key}'].OutputValue | [0]" \
    --output text 2>/dev/null || true
}

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" || "$value" == "None" ]]; then
    echo "[start-training] Missing CloudFormation output: $name" >&2
    exit 1
  fi
}

print_prerequisite_hint() {
  cat >&2 <<EOF
[start-training] Training EC2 host is not available yet.
[start-training] Expected output key: TrainingEc2InstanceId (stack: ${TRAINING_STACK_NAME})

[start-training] Quick checks:
  aws cloudformation list-stack-resources --stack-name ${TRAINING_STACK_NAME} --region ${AWS_REGION}
  aws ec2 describe-instances --instance-ids i-xxxxxxxxxxxxxxxxx --region ${AWS_REGION}

[start-training] You can bypass stack lookup by passing an explicit instance id:
  TRAINING_INSTANCE_ID=i-xxxxxxxxxxxxxxxxx bash scripts/start-training.sh
EOF
}

TRAINING_INSTANCE_ID="${TRAINING_INSTANCE_ID:-$(get_output "$TRAINING_STACK_NAME" "TrainingEc2InstanceId")}" 
TRAINING_LOG_GROUP_NAME="${TRAINING_LOG_GROUP_NAME:-$(get_output "$TRAINING_STACK_NAME" "TrainingEc2LogGroupName")}" 
TRAINING_IMAGE_URI="${TRAINING_IMAGE_URI:-$(get_output "$STACK_NAME" "TrainingImageUri")}" 
PARQUET_DATA_BUCKET="${PARQUET_DATA_BUCKET:-$(get_output "$STACK_NAME" "ParquetDataBucketName")}" 
MODEL_BUCKET="${MODEL_BUCKET:-$(get_output "$STACK_NAME" "MlModelBucketName")}" 

if [[ -z "$TRAINING_INSTANCE_ID" || "$TRAINING_INSTANCE_ID" == "None" ]]; then
  print_prerequisite_hint
  exit 1
fi

if [[ -z "$TRAINING_LOG_GROUP_NAME" || "$TRAINING_LOG_GROUP_NAME" == "None" ]]; then
  TRAINING_LOG_GROUP_NAME="/aetherforecast/training-ec2-manual"
  echo "[start-training] Training log group output not found. Using default hint: ${TRAINING_LOG_GROUP_NAME}"
fi

require_value "TrainingImageUri" "$TRAINING_IMAGE_URI"
require_value "ParquetDataBucketName" "$PARQUET_DATA_BUCKET"
require_value "MlModelBucketName" "$MODEL_BUCKET"

MODEL_S3_URI="${MODEL_S3_URI:-s3://${MODEL_BUCKET}/chronos-v1/model/}"
CHECKPOINT_S3_URI="${CHECKPOINT_S3_URI:-s3://${MODEL_BUCKET}/checkpoints/}"

INSTANCE_READY_TO_STOP="false"
cleanup() {
  if [[ "$INSTANCE_READY_TO_STOP" == "true" && "$KEEP_INSTANCE_RUNNING" != "true" ]]; then
    echo "[start-training] Stopping training instance $TRAINING_INSTANCE_ID"
    aws ec2 stop-instances --instance-ids "$TRAINING_INSTANCE_ID" --region "$AWS_REGION" >/dev/null
    aws ec2 wait instance-stopped --instance-ids "$TRAINING_INSTANCE_ID" --region "$AWS_REGION"
    echo "[start-training] Training instance stopped"
  fi
}
trap cleanup EXIT

INSTANCE_STATE="$(aws ec2 describe-instances --instance-ids "$TRAINING_INSTANCE_ID" --region "$AWS_REGION" --query "Reservations[0].Instances[0].State.Name" --output text)"

if [[ "$INSTANCE_STATE" != "running" ]]; then
  echo "[start-training] Starting training instance $TRAINING_INSTANCE_ID"
  aws ec2 start-instances --instance-ids "$TRAINING_INSTANCE_ID" --region "$AWS_REGION" >/dev/null
  aws ec2 wait instance-running --instance-ids "$TRAINING_INSTANCE_ID" --region "$AWS_REGION"
else
  echo "[start-training] Training instance already running"
fi

INSTANCE_READY_TO_STOP="true"

echo "[start-training] Waiting for SSM agent to become Online"
for ((attempt=1; attempt<=SSM_WAIT_ATTEMPTS; attempt++)); do
  ping_status="$(aws ssm describe-instance-information --region "$AWS_REGION" --filters "Key=InstanceIds,Values=${TRAINING_INSTANCE_ID}" --query "InstanceInformationList[0].PingStatus" --output text 2>/dev/null || true)"
  if [[ "$ping_status" == "Online" ]]; then
    echo "[start-training] SSM Online"
    break
  fi

  if [[ "$attempt" -eq "$SSM_WAIT_ATTEMPTS" ]]; then
    echo "[start-training] Timeout waiting for SSM Online" >&2
    exit 1
  fi

  sleep "$SSM_WAIT_INTERVAL_SECONDS"
done

PARAM_FILE="$(mktemp)"
cat > "$PARAM_FILE" <<EOF
{
  "commands": [
    "set -euo pipefail",
    "export TRAIN_IMAGE_URI='${TRAINING_IMAGE_URI}'",
    "export AWS_REGION='${AWS_REGION}'",
    "export DATA_S3_BUCKET='${PARQUET_DATA_BUCKET}'",
    "export MODEL_S3_URI='${MODEL_S3_URI}'",
    "export CHECKPOINT_S3_URI='${CHECKPOINT_S3_URI}'",
    "export SYMBOLS='${SYMBOLS}'",
    "export TIMEFRAME='${TIMEFRAME}'",
    "export TRAINING_HORIZON='${TRAINING_HORIZON}'",
    "export CONTEXT_LENGTH='${CONTEXT_LENGTH}'",
    "export MAX_ROWS_PER_SYMBOL='${MAX_ROWS_PER_SYMBOL}'",
    "export EPOCHS='${EPOCHS}'",
    "export LEARNING_RATE='${LEARNING_RATE}'",
    "export BATCH_SIZE='${BATCH_SIZE}'",
    "export GRAD_ACCUM_STEPS='${GRAD_ACCUM_STEPS}'",
    "export WARMUP_RATIO='${WARMUP_RATIO}'",
    "export WEIGHT_DECAY='${WEIGHT_DECAY}'",
    "export SAVE_STEPS='${SAVE_STEPS}'",
    "export EVAL_STEPS='${EVAL_STEPS}'",
    "export LOGGING_STEPS='${LOGGING_STEPS}'",
    "export LORA_R='${LORA_R}'",
    "export LORA_ALPHA='${LORA_ALPHA}'",
    "export LORA_DROPOUT='${LORA_DROPOUT}'",
    "export MAX_SEQ_LENGTH='${MAX_SEQ_LENGTH}'",
    "export BASE_MODEL_ID='${BASE_MODEL_ID}'",
    "export BASE_MODEL_FALLBACK_ID='${BASE_MODEL_FALLBACK_ID}'",
    "if [ -x /opt/aetherforecast-training/run-training.sh ]; then",
    "  sudo -E /opt/aetherforecast-training/run-training.sh",
    "else",
    "  command -v docker >/dev/null 2>&1 || { echo '[start-training] docker is not installed on training host'; exit 65; }",
    "  sudo systemctl start docker >/dev/null 2>&1 || true",
    "  ECR_REGISTRY=$(echo \"$TRAIN_IMAGE_URI\" | cut -d'/' -f1)",
    "  if echo \"$ECR_REGISTRY\" | grep -q '\\\\.dkr\\\\.ecr\\\\.'; then aws ecr get-login-password --region \"$AWS_REGION\" | sudo docker login --username AWS --password-stdin \"$ECR_REGISTRY\"; fi",
    "  sudo docker pull \"$TRAIN_IMAGE_URI\"",
    "  sudo docker rm -f aetherforecast-training-run >/dev/null 2>&1 || true",
    "  sudo docker run --name aetherforecast-training-run --rm --gpus all -e TRAIN_MODE=true -e AWS_REGION=\"$AWS_REGION\" -e DATA_S3_BUCKET=\"$DATA_S3_BUCKET\" -e DATA_BUCKET=\"$DATA_S3_BUCKET\" -e MODEL_S3_URI=\"$MODEL_S3_URI\" -e CHECKPOINT_S3_URI=\"$CHECKPOINT_S3_URI\" -e SYMBOLS=\"$SYMBOLS\" -e TIMEFRAME=\"$TIMEFRAME\" -e TRAINING_HORIZON=\"$TRAINING_HORIZON\" -e CONTEXT_LENGTH=\"$CONTEXT_LENGTH\" -e MAX_ROWS_PER_SYMBOL=\"$MAX_ROWS_PER_SYMBOL\" -e EPOCHS=\"$EPOCHS\" -e LEARNING_RATE=\"$LEARNING_RATE\" -e BATCH_SIZE=\"$BATCH_SIZE\" -e GRAD_ACCUM_STEPS=\"$GRAD_ACCUM_STEPS\" -e WARMUP_RATIO=\"$WARMUP_RATIO\" -e WEIGHT_DECAY=\"$WEIGHT_DECAY\" -e SAVE_STEPS=\"$SAVE_STEPS\" -e EVAL_STEPS=\"$EVAL_STEPS\" -e LOGGING_STEPS=\"$LOGGING_STEPS\" -e LORA_R=\"$LORA_R\" -e LORA_ALPHA=\"$LORA_ALPHA\" -e LORA_DROPOUT=\"$LORA_DROPOUT\" -e MAX_SEQ_LENGTH=\"$MAX_SEQ_LENGTH\" -e BASE_MODEL_ID=\"$BASE_MODEL_ID\" -e BASE_MODEL_FALLBACK_ID=\"$BASE_MODEL_FALLBACK_ID\" -e HF_HOME=/tmp/hf-home -e HF_CACHE_DIR=/tmp/hf-cache \"$TRAIN_IMAGE_URI\"",
    "fi"
  ]
}
EOF

COMMAND_ID="$(aws ssm send-command \
  --instance-ids "$TRAINING_INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "AetherForecast manual training run" \
  --parameters "file://${PARAM_FILE}" \
  --region "$AWS_REGION" \
  --query "Command.CommandId" \
  --output text)"

rm -f "$PARAM_FILE"

echo "[start-training] SSM command submitted: $COMMAND_ID"
echo "[start-training] CloudWatch logs: aws logs tail ${TRAINING_LOG_GROUP_NAME} --region ${AWS_REGION} --follow --since 15m"

FINAL_STATUS="Pending"
while true; do
  FINAL_STATUS="$(aws ssm get-command-invocation --command-id "$COMMAND_ID" --instance-id "$TRAINING_INSTANCE_ID" --region "$AWS_REGION" --query "Status" --output text 2>/dev/null || true)"

  if [[ "$FINAL_STATUS" == "Success" || "$FINAL_STATUS" == "Failed" || "$FINAL_STATUS" == "TimedOut" || "$FINAL_STATUS" == "Cancelled" ]]; then
    break
  fi

  echo "[start-training] Training status: $FINAL_STATUS"
  sleep "$COMMAND_WAIT_INTERVAL_SECONDS"
done

STDOUT_CONTENT="$(aws ssm get-command-invocation --command-id "$COMMAND_ID" --instance-id "$TRAINING_INSTANCE_ID" --region "$AWS_REGION" --query "StandardOutputContent" --output text || true)"
STDERR_CONTENT="$(aws ssm get-command-invocation --command-id "$COMMAND_ID" --instance-id "$TRAINING_INSTANCE_ID" --region "$AWS_REGION" --query "StandardErrorContent" --output text || true)"

echo "[start-training] Final status: $FINAL_STATUS"
if [[ -n "$STDOUT_CONTENT" && "$STDOUT_CONTENT" != "None" ]]; then
  echo "----- SSM STDOUT -----"
  echo "$STDOUT_CONTENT"
fi
if [[ -n "$STDERR_CONTENT" && "$STDERR_CONTENT" != "None" ]]; then
  echo "----- SSM STDERR -----"
  echo "$STDERR_CONTENT"
fi

if [[ "$FINAL_STATUS" != "Success" ]]; then
  echo "[start-training] Training failed with status $FINAL_STATUS" >&2
  exit 1
fi

echo "[start-training] Training completed successfully"
