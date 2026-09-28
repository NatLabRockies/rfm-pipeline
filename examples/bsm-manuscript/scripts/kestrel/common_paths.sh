#!/usr/bin/env bash
# Shared path resolution helpers for Kestrel operational scripts.

kestrel_default_artifacts_root() {
  local repo_root="$1"
  local scratch_artifacts_root="/scratch/${USER}/bsm/bsm-public-rf/artifacts"
  if [[ -d "${scratch_artifacts_root}" || -L "${scratch_artifacts_root}" ]]; then
    printf '%s\n' "${scratch_artifacts_root}"
    return 0
  fi
  printf '%s\n' "${repo_root}/artifacts"
}

kestrel_default_logs_root() {
  printf '%s\n' "/scratch/${USER}/bsm"
}

kestrel_require_non_home_path() {
  local path_value="$1"
  local label="$2"
  local user_name="${USER:-}"
  if [[ -n "${user_name}" && "${path_value}" == "/home/${user_name}/"* ]]; then
    echo "error: ${label} cannot be under /home/${user_name}: ${path_value}" >&2
    return 1
  fi
  return 0
}
