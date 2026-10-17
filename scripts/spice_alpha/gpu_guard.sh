# Source from an sbatch script: on nodes that expose every GPU to every job while
# setting CUDA_VISIBLE_DEVICES=0 (seen on sgd01), claim an idle, unclaimed GPU under
# a per-node lock instead of piling onto card 0. Claims are released on exit.
if command -v nvidia-smi >/dev/null 2>&1; then
  n_visible="$(nvidia-smi -L | wc -l)"
  if [[ "$n_visible" -gt 1 ]]; then
    # Pick under a per-node lock and record the claim, so jobs that start at
    # the same moment (before either has allocated memory) cannot pick the same card.
    claim_dir="$HOME/.cache/gpu_guard/$(hostname)"; mkdir -p "$claim_dir"
    exec 9>"$claim_dir/.lock"; flock 9
    for f in "$claim_dir"/*.claim; do
      [[ -e "$f" ]] || continue
      squeue -h -j "$(basename "$f" .claim)" -t R 2>/dev/null | grep -q . || rm -f "$f"
    done
    claimed="$(cat "$claim_dir"/*.claim 2>/dev/null | sort -u || true)"
    pick=""
    while IFS=, read -r idx uuid used; do
      uuid="${uuid// /}"; used="${used// /}"
      grep -qx "$uuid" <<<"$claimed" && continue
      [[ "$used" -gt 1000 ]] && continue
      pick="$idx"; pick_uuid="$uuid"; break
    done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
    if [[ -n "$pick" ]]; then
      [[ "$pick" != "${CUDA_VISIBLE_DEVICES:-0}" ]] && echo "GPU guard: using GPU $pick ($pick_uuid; ${n_visible} visible)" >&2
      export CUDA_VISIBLE_DEVICES="$pick"
      echo "$pick_uuid" > "$claim_dir/${SLURM_JOB_ID}.claim"
    else
      echo "GPU guard: no unclaimed idle GPU among ${n_visible}; keeping ${CUDA_VISIBLE_DEVICES:-0}" >&2
    fi
    flock -u 9; exec 9>&-
    trap 'rm -f "$HOME/.cache/gpu_guard/$(hostname)/${SLURM_JOB_ID}.claim"' EXIT
  fi
fi
