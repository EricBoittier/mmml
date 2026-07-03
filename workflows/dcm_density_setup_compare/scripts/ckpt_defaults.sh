# Shared MMML_CKPT fallback for dcm_density_setup_compare (source, do not execute).
default_mmml_ckpt() {
  local repo_root="$1"
  local candidates=(
    "/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts/dcm1/dcm1_params.json"
    "${repo_root}/examples/ckpts_json/DESdimers_params.json"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  echo "${candidates[1]}"
}
