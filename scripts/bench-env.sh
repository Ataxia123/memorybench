#!/usr/bin/env bash
# bench-env.sh — export BONFIRES_API_KEY and OPENROUTER_API_KEY from delve/.env
#
# Usage:
#   source scripts/bench-env.sh            # export keys into the current shell
#   ./scripts/bench-env.sh <command...>    # run a command with the keys exported
#
# SECURITY: key values are NEVER echoed; only "KEY len=NN" confirmations.
# Server-side HYPERMEM knobs live in memory_kernel/config/phase1-bench.env
# (applied via delve/scripts/restart_uvicorn_lean.py --env-file), not here.

BENCH_ENV_DELVE_ENV="${BENCH_ENV_DELVE_ENV:-/home/at0x/Vaults/Bonfires/delve/.env}"

# Detect whether we are being sourced (works in bash and zsh).
_bench_env_sourced=0
if (return 0 2>/dev/null); then
  _bench_env_sourced=1
fi

_bench_env_fail() {
  echo "bench-env.sh ERROR: $1" >&2
  if [ "${_bench_env_sourced}" -eq 1 ]; then
    return 1
  else
    exit 1
  fi
}

# Read the last assignment of a key from delve/.env; strips optional
# `export ` prefix and surrounding single/double quotes. Never echoes values
# anywhere except the captured stdout of this function.
_bench_env_read_key() {
  local key="$1" line val
  line=$(grep -E "^(export[[:space:]]+)?${key}=" "${BENCH_ENV_DELVE_ENV}" | tail -1)
  [ -n "${line}" ] || return 1
  val="${line#*"${key}"=}"
  val="${val%$'\r'}"
  case "${val}" in
    \"*\") val="${val#\"}"; val="${val%\"}" ;;
    \'*\') val="${val#\'}"; val="${val%\'}" ;;
  esac
  [ -n "${val}" ] || return 1
  printf '%s' "${val}"
}

bench_env_load() {
  if [ ! -f "${BENCH_ENV_DELVE_ENV}" ]; then
    _bench_env_fail "delve env file not found: ${BENCH_ENV_DELVE_ENV}" || return 1
  fi

  local _missing=0 _key _val
  for _key in BONFIRES_API_KEY OPENROUTER_API_KEY; do
    if _val=$(_bench_env_read_key "${_key}"); then
      export "${_key}=${_val}"
      echo "bench-env: ${_key} len=${#_val}"
    else
      echo "bench-env.sh ERROR: ${_key} missing or empty in ${BENCH_ENV_DELVE_ENV}" >&2
      _missing=1
    fi
  done
  unset _val

  if [ "${_missing}" -ne 0 ]; then
    _bench_env_fail "required keys missing from ${BENCH_ENV_DELVE_ENV}" || return 1
  fi
  return 0
}

bench_env_load || { [ "${_bench_env_sourced}" -eq 1 ] && return 1 || exit 1; }

if [ "${_bench_env_sourced}" -eq 0 ] && [ "$#" -gt 0 ]; then
  exec "$@"
fi
