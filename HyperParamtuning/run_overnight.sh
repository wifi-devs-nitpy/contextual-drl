#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the project root, or pass PROJECT_DIR=/path/to/project.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/overnight}"
mkdir -p "${OUTPUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/overnight_${STAMP}.log"
STATUS_FILE="${OUTPUT_DIR}/overnight_${STAMP}.status"

source "C:/Users/jomon/Documents/wifi_9/mapc-cmab/HyperParamtuning/exports.sh"
echo "cheking for the exported variables"
echo "$SMTP_USER"
echo "$SMTP_PASSWORD"
echo "$EMAIL_TO"

cd "${PROJECT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

START_TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
echo "Hierarchical DQN overnight experiment"
echo "Started: ${START_TS}"
echo "Project: ${PROJECT_DIR}"
echo "Log: ${LOG_FILE}"
echo "============================================================"

# Recommended: export SMTP_* and EMAIL_TO before launching.
# Example variables are documented at the bottom of this file.

send_email() {
    local subject="$1"
    local body="$2"

    if [[ -n "${SMTP_HOST:-}" && -n "${SMTP_USER:-}" && -n "${SMTP_PASSWORD:-}" && -n "${EMAIL_TO:-}" ]]; then
        "${PYTHON_BIN}" - "${subject}" "${body}" <<'PY'
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path

subject, body = sys.argv[1], sys.argv[2]
msg = EmailMessage()
msg["From"] = os.environ["SMTP_USER"]
msg["To"] = os.environ["EMAIL_TO"]
msg["Subject"] = subject
msg.set_content(body)

host = os.environ["SMTP_HOST"]
port = int(os.environ.get("SMTP_PORT", "587"))
use_ssl = os.environ.get("SMTP_SSL", "0") == "1"

if use_ssl:
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as smtp:
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
else:
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
PY
        echo "Email sent to ${EMAIL_TO}"
        return 0
    fi

    if command -v mail >/dev/null 2>&1 && [[ -n "${EMAIL_TO:-}" ]]; then
        printf '%s\n' "${body}" | mail -s "${subject}" "${EMAIL_TO}"
        echo "Email sent using local mail command to ${EMAIL_TO}"
        return 0
    fi

    echo "Email not sent: configure SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_TO."
    return 0
}

on_error() {
    local exit_code=$?
    local end_ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    local tail="$(tail -n 80 "${LOG_FILE}" 2>/dev/null || true)"
    printf 'FAILED\nStarted: %s\nEnded: %s\nExit code: %s\n\nLast log lines:\n%s\n' \
        "${START_TS}" "${end_ts}" "${exit_code}" "${tail}" > "${STATUS_FILE}"
    send_email "MAPC DQN overnight FAILED" "$(cat "${STATUS_FILE}")" || true
    exit "${exit_code}"
}

on_success() {
    local end_ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    {
        echo "SUCCESS"
        echo "Started: ${START_TS}"
        echo "Ended: ${end_ts}"
        echo "Output: ${OUTPUT_DIR}"
        echo
        if [[ -f "${OUTPUT_DIR}/sweep_ranking.txt" ]]; then
            echo "Sweep ranking:"
            cat "${OUTPUT_DIR}/sweep_ranking.txt"
            echo
        fi
        if [[ -f "${OUTPUT_DIR}/final_metrics.json" ]]; then
            echo "Final metrics:"
            cat "${OUTPUT_DIR}/final_metrics.json"
        fi
    } > "${STATUS_FILE}"
    send_email "MAPC DQN overnight SUCCEEDED" "$(cat "${STATUS_FILE}")" || true
}

trap on_error ERR

"${PYTHON_BIN}" -u tune_hierarchical_dqn.py \
    --mode all \
    --sweep-runs "${SWEEP_RUNS:-3}" \
    --sweep-steps "${SWEEP_STEPS:-2500}" \
    --final-runs "${FINAL_RUNS:-20}" \
    --final-steps "${FINAL_STEPS:-10000}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${JAX_CACHE_DIR:-${PROJECT_DIR}/jax_cache_overnight}"

on_success

echo "============================================================"
echo "Finished successfully."
echo "Log: ${LOG_FILE}"
echo "Status: ${STATUS_FILE}"
echo "============================================================"

# Gmail example:
# export SMTP_HOST=smtp.gmail.com
# export SMTP_PORT=587
# export SMTP_USER='youraddress@gmail.com'
# export SMTP_PASSWORD='your-16-character-app-password'
# export EMAIL_TO='youraddress@gmail.com'
# ./run_overnight.sh
