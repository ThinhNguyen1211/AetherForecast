#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-southeast-1}"
STACK_NAME="${STACK_NAME:-AetherForecastStack}"
TRAINING_STACK_NAME="${TRAINING_STACK_NAME:-$STACK_NAME}"

get_output() {
  local stack_name="$1"
  local key="$2"
  aws cloudformation describe-stacks \
    --stack-name "$stack_name" \
    --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='${key}'].OutputValue | [0]" \
    --output text 2>/dev/null || true
}

TRAINING_INSTANCE_ID="${TRAINING_INSTANCE_ID:-$(get_output "$TRAINING_STACK_NAME" "TrainingEc2InstanceId")}" 
if [[ -z "$TRAINING_INSTANCE_ID" || "$TRAINING_INSTANCE_ID" == "None" ]]; then
  echo "[stop-training] Missing TrainingEc2InstanceId in stack: ${TRAINING_STACK_NAME}" >&2
  echo "[stop-training] Provide explicit instance id if host exists: TRAINING_INSTANCE_ID=i-xxxxxxxxxxxxxxxxx bash scripts/stop-training.sh" >&2
  exit 1
fi

STATE="$(aws ec2 describe-instances --instance-ids "$TRAINING_INSTANCE_ID" --region "$AWS_REGION" --query "Reservations[0].Instances[0].State.Name" --output text)"
if [[ "$STATE" == "stopped" ]]; then
  echo "[stop-training] Instance already stopped: $TRAINING_INSTANCE_ID"
  exit 0
fi

echo "[stop-training] Stopping instance $TRAINING_INSTANCE_ID"
aws ec2 stop-instances --instance-ids "$TRAINING_INSTANCE_ID" --region "$AWS_REGION" >/dev/null
aws ec2 wait instance-stopped --instance-ids "$TRAINING_INSTANCE_ID" --region "$AWS_REGION"
echo "[stop-training] Instance stopped"
