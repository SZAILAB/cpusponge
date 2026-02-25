#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP_PATH="${ZIP_PATH:-${ROOT_DIR}/test.zip}"
APPENDIX_OUT="${APPENDIX_OUT:-${ROOT_DIR}/reports/step5000_appendix}"
GPU_BIN="${GPU_BIN:-/home/wuping/SPONGE/SPONGE}"

CASE_NAME="${CASE_NAME:-mdin_npt_andersenbaro}"
STEP_LIMIT="${STEP_LIMIT:-5000}"
WRITE_INTERVAL="${WRITE_INFORMATION_INTERVAL:-100}"
GPU_RUNS="${GPU_RUNS:-10}"
GPU_TIMEOUT_SECONDS="${GPU_TIMEOUT_SECONDS:-7200}"
FORCE_RERUN="${FORCE_RERUN:-0}"

log() {
    printf '[appendix] %s\n' "$*"
}

die() {
    printf '[appendix][ERROR] %s\n' "$*" >&2
    exit 1
}

ensure_tools() {
    command -v unzip >/dev/null 2>&1 || die "unzip is required"
    command -v awk >/dev/null 2>&1 || die "awk is required"
    command -v timeout >/dev/null 2>&1 || die "timeout is required"
    command -v rg >/dev/null 2>&1 || die "rg is required"
    [[ -x "${GPU_BIN}" ]] || die "GPU binary not executable: ${GPU_BIN}"
    [[ "${GPU_RUNS}" =~ ^[0-9]+$ ]] || die "GPU_RUNS must be a positive integer"
    [[ "${GPU_RUNS}" -ge 1 ]] || die "GPU_RUNS must be >= 1"
}

prepare_dirs() {
    mkdir -p "${APPENDIX_OUT}/cases" "${APPENDIX_OUT}/mdin" "${APPENDIX_OUT}/mdout" \
        "${APPENDIX_OUT}/mdinfo" "${APPENDIX_OUT}/logs"
}

extract_case() {
    local entry="test/${CASE_NAME}.txt"
    unzip -p "${ZIP_PATH}" "${entry}" > "${APPENDIX_OUT}/cases/${CASE_NAME}.orig.mdin" \
        || die "failed to extract ${entry} from ${ZIP_PATH}"
}

rewrite_mdin() {
    local input_mdin="$1"
    local output_mdin="$2"
    local mdout_path="$3"
    local mdinfo_path="$4"

    awk -v step_limit="${STEP_LIMIT}" \
        -v write_interval="${WRITE_INTERVAL}" \
        -v mdout="${mdout_path}" \
        -v mdinfo="${mdinfo_path}" '
        function trim(s){gsub(/^[ \t]+|[ \t]+$/, "", s); return s}
        BEGIN{seen_step=seen_wi=seen_prefix=seen_coord=seen_vel=seen_mdout=seen_mdinfo=0}
        {
            line=$0
            if (index(line, "=") > 0) {
                split(line, a, "=")
                key=tolower(trim(a[1]))
                if (key=="step_limit") {print "step_limit = " step_limit; seen_step=1; next}
                if (key=="write_information_interval") {print "write_information_interval = " write_interval; seen_wi=1; next}
                if (key=="default_in_file_prefix") {print "default_in_file_prefix = /home/wuping/SPONGE/covid-tip4p/covid-tip4p"; seen_prefix=1; next}
                if (key=="coordinate_in_file") {print "coordinate_in_file = /home/wuping/SPONGE/covid-tip4p/npt_coordinate.txt"; seen_coord=1; next}
                if (key=="velocity_in_file") {print "velocity_in_file = /home/wuping/SPONGE/covid-tip4p/npt_velocity.txt"; seen_vel=1; next}
                if (key=="mdout") {print "mdout = " mdout; seen_mdout=1; next}
                if (key=="mdinfo") {print "mdinfo = " mdinfo; seen_mdinfo=1; next}
            }
            print line
        }
        END{
            if(!seen_step) print "step_limit = " step_limit
            if(!seen_wi) print "write_information_interval = " write_interval
            if(!seen_prefix) print "default_in_file_prefix = /home/wuping/SPONGE/covid-tip4p/covid-tip4p"
            if(!seen_coord) print "coordinate_in_file = /home/wuping/SPONGE/covid-tip4p/npt_coordinate.txt"
            if(!seen_vel) print "velocity_in_file = /home/wuping/SPONGE/covid-tip4p/npt_velocity.txt"
            if(!seen_mdout) print "mdout = " mdout
            if(!seen_mdinfo) print "mdinfo = " mdinfo
        }
    ' "${input_mdin}" > "${output_mdin}"
}

scan_nonfinite_in_log() {
    local log_file="$1"
    if rg -i -n -m 1 'non[- ]finite|(^|[^A-Za-z0-9_])(nan|inf)([^A-Za-z0-9_]|$)|spongeerror|SPONGE_ERROR' "${log_file}" >/dev/null 2>&1; then
        echo "1"
    else
        echo "0"
    fi
}

last_step_from_mdout() {
    local mdout="$1"
    if [[ ! -s "${mdout}" ]]; then
        echo "N/A"
        return
    fi
    awk '
        NR == 1 {next}
        {
            if ($1 ~ /^[0-9]+$/) last = $1
        }
        END {
            if (last == "") print "N/A"; else print last
        }
    ' "${mdout}"
}

run_gpu_repeats() {
    local input_mdin="${APPENDIX_OUT}/cases/${CASE_NAME}.orig.mdin"
    local run_idx run_id tag mdin_path mdout_path mdinfo_path log_path

    echo "run_id,exit_code,elapsed_s,last_step,nonfinite_flag,log_path,mdout_path,mdinfo_path,mdin_path" > "${APPENDIX_OUT}/gpu_repeat_index.csv"

    for ((run_idx=1; run_idx<=GPU_RUNS; run_idx++)); do
        run_id="$(printf "run_%03d" "${run_idx}")"
        tag="${CASE_NAME}.gpu_ref.${run_id}"
        mdin_path="${APPENDIX_OUT}/mdin/${tag}.mdin"
        mdout_path="${APPENDIX_OUT}/mdout/${tag}.mdout.txt"
        mdinfo_path="${APPENDIX_OUT}/mdinfo/${tag}.mdinfo.txt"
        log_path="${APPENDIX_OUT}/logs/${tag}.log"

        if [[ "${FORCE_RERUN}" != "1" && -s "${mdout_path}" ]]; then
            log "skip existing ${tag}"
            echo "${run_id},0,0,$(last_step_from_mdout "${mdout_path}"),$(scan_nonfinite_in_log "${log_path}"),${log_path},${mdout_path},${mdinfo_path},${mdin_path}" >> "${APPENDIX_OUT}/gpu_repeat_index.csv"
            continue
        fi

        rewrite_mdin "${input_mdin}" "${mdin_path}" "${mdout_path}" "${mdinfo_path}"

        log "run ${tag}"
        local t0 t1 elapsed exit_code
        t0="$(date +%s)"
        set +e
        timeout "${GPU_TIMEOUT_SECONDS}s" "${GPU_BIN}" -mdin "${mdin_path}" > "${log_path}" 2>&1
        exit_code="$?"
        set -e
        t1="$(date +%s)"
        elapsed="$((t1 - t0))"

        echo "${run_id},${exit_code},${elapsed},$(last_step_from_mdout "${mdout_path}"),$(scan_nonfinite_in_log "${log_path}"),${log_path},${mdout_path},${mdinfo_path},${mdin_path}" >> "${APPENDIX_OUT}/gpu_repeat_index.csv"
    done
}

main() {
    ensure_tools
    prepare_dirs
    extract_case
    run_gpu_repeats
    log "gpu repeats complete: ${APPENDIX_OUT}/gpu_repeat_index.csv"
}

main "$@"
