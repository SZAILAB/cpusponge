#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP_PATH="${ZIP_PATH:-${ROOT_DIR}/test.zip}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/reports/step1000}"
STAGING_ROOT="${STAGING_ROOT:-/tmp/cpu_sponge_step1000_stage}"
GPU_BIN="${GPU_BIN:-/home/wuping/SPONGE/SPONGE}"

STEP_LIMIT="${STEP_LIMIT:-1000}"
WRITE_INTERVAL="${WRITE_INFORMATION_INTERVAL:-100}"
CPU_TIMEOUT_SECONDS="${CPU_TIMEOUT_SECONDS:-14400}"
GPU_TIMEOUT_SECONDS="${GPU_TIMEOUT_SECONDS:-3600}"
FORCE_RERUN="${FORCE_RERUN:-0}"
PARALLEL_BACKENDS="${PARALLEL_BACKENDS:-1}"
ERR_STEP_START="${ERR_STEP_START:-${WRITE_INTERVAL}}"
ERR_STEP_INTERVAL="${ERR_STEP_INTERVAL:-${WRITE_INTERVAL}}"
ERR_STEP_END="${ERR_STEP_END:-${STEP_LIMIT}}"
BACKENDS_CSV="${BACKENDS_CSV:-}"
CASES_CSV="${CASES_CSV:-}"
REPEAT_CASE="${REPEAT_CASE:-mdin_npt_mcbaro}"
SKIP_REPEAT="${SKIP_REPEAT:-0}"

BACKENDS=(gpu_ref local_cpu)
CPU_BACKENDS=(local_cpu)
CASES=()

log() {
    printf '[step1000] %s\n' "$*"
}

die() {
    printf '[step1000][ERROR] %s\n' "$*" >&2
    exit 1
}

ensure_tools() {
    command -v awk >/dev/null 2>&1 || die "awk is required"
    command -v unzip >/dev/null 2>&1 || die "unzip is required"
    command -v sed >/dev/null 2>&1 || die "sed is required"
    command -v timeout >/dev/null 2>&1 || die "timeout is required"
    command -v flock >/dev/null 2>&1 || die "flock is required"
    command -v rsync >/dev/null 2>&1 || die "rsync is required"

    [[ "${PARALLEL_BACKENDS}" =~ ^[0-9]+$ ]] || die "PARALLEL_BACKENDS must be a positive integer"
    [[ "${PARALLEL_BACKENDS}" -ge 1 ]] || die "PARALLEL_BACKENDS must be >= 1"
    [[ "${ERR_STEP_START}" =~ ^[0-9]+$ ]] || die "ERR_STEP_START must be a positive integer"
    [[ "${ERR_STEP_INTERVAL}" =~ ^[0-9]+$ ]] || die "ERR_STEP_INTERVAL must be a positive integer"
    [[ "${ERR_STEP_END}" =~ ^[0-9]+$ ]] || die "ERR_STEP_END must be a positive integer"
    [[ "${ERR_STEP_START}" -ge 1 ]] || die "ERR_STEP_START must be >= 1"
    [[ "${ERR_STEP_INTERVAL}" -ge 1 ]] || die "ERR_STEP_INTERVAL must be >= 1"
    [[ "${ERR_STEP_END}" -ge "${ERR_STEP_START}" ]] || die "ERR_STEP_END must be >= ERR_STEP_START"
}

required_modules=(
    Lennard_Jones_force MD_core No_PBC PME_force SITS angle barostats bias
    bond cmap collective_variable constrain dihedral nb14 neighbor_list
    plugin restrain thermostats virtual_atoms wall
)

supported_backends=(gpu_ref local_cpu)

is_supported_backend() {
    local backend="$1"
    local b
    for b in "${supported_backends[@]}"; do
        if [[ "${b}" == "${backend}" ]]; then
            return 0
        fi
    done
    return 1
}

configure_backends() {
    if [[ -n "${BACKENDS_CSV}" ]]; then
        IFS=',' read -r -a BACKENDS <<< "${BACKENDS_CSV}"
    fi
    [[ "${#BACKENDS[@]}" -gt 0 ]] || die "BACKENDS is empty"

    CPU_BACKENDS=()
    local backend
    for backend in "${BACKENDS[@]}"; do
        is_supported_backend "${backend}" || die "unsupported backend: ${backend}"
        if [[ "${backend}" != "gpu_ref" ]]; then
            CPU_BACKENDS+=("${backend}")
        fi
    done
}

backend_source_dir() {
    local backend="$1"
    if [[ "${backend}" == "gpu_ref" ]]; then
        printf '%s\n' "/home/wuping/SPONGE"
        return
    fi
    if [[ "${backend}" == "local_cpu" ]]; then
        printf '%s\n' "${STAGING_ROOT}/src/local_cpu"
        return
    fi
    printf '%s\n' "${STAGING_ROOT}/src/${backend}"
}

backend_binary() {
    local backend="$1"
    case "${backend}" in
        gpu_ref)
            printf '%s\n' "${GPU_BIN}"
            ;;
        local_cpu)
            printf '%s\n' "${STAGING_ROOT}/src/local_cpu/sponge_cpu_mainpath"
            ;;
        *)
            die "unknown backend: ${backend}"
            ;;
    esac
}

normalize_line_endings_if_needed() {
    local backend="$1"
    local source_dir
    source_dir="$(backend_source_dir "${backend}")"
    if [[ "${backend}" == "local_cpu" ]]; then
        if [[ -d "${source_dir}/covid-tip4p" ]]; then
            sed -i 's/\r$//' "${source_dir}"/covid-tip4p/* || true
        fi
    fi
}

prepare_directories() {
    mkdir -p "${OUT_DIR}" "${OUT_DIR}/cases" "${OUT_DIR}/mdin" "${OUT_DIR}/logs" \
        "${OUT_DIR}/mdout" "${OUT_DIR}/mdinfo" "${OUT_DIR}/metrics" "${OUT_DIR}/errors" \
        "${OUT_DIR}/build_logs"
    mkdir -p "${STAGING_ROOT}/src"
}

copy_sources_to_staging() {
    local backend
    for backend in "${CPU_BACKENDS[@]}"; do
        local src
        if [[ "${backend}" == "local_cpu" ]]; then
            src="${ROOT_DIR}"
        else
            src="${ROOT_DIR}/${backend}"
        fi
        local dst="${STAGING_ROOT}/src/${backend}"
        [[ -d "${src}" ]] || die "source dir missing: ${src}"
        if [[ ! -d "${dst}" ]]; then
            log "copy source to staging: ${backend}"
            if [[ "${backend}" == "local_cpu" ]]; then
                mkdir -p "${dst}"
                rsync -a --delete --exclude '.git/' --exclude 'reports/' --exclude 'test.zip' --exclude 'tools/' "${src}/" "${dst}/"
            else
                cp -a "${src}" "${dst}"
            fi
        fi
    done
}

build_backend() {
    local backend="$1"
    local source_dir
    source_dir="$(backend_source_dir "${backend}")"
    local build_log="${OUT_DIR}/build_logs/${backend}.build.log"

    normalize_line_endings_if_needed "${backend}"

    log "build ${backend}"
    case "${backend}" in
        local_cpu)
            (cd "${source_dir}" && make -f Makefile.cpu -j"$(nproc)" sponge_cpu_mainpath) >"${build_log}" 2>&1
            ;;
        *)
            die "build not supported for backend: ${backend}"
            ;;
    esac

    local bin
    bin="$(backend_binary "${backend}")"
    [[ -x "${bin}" ]] || die "backend binary is not executable after build: ${bin}"
}

extract_cases_from_zip() {
    [[ -f "${ZIP_PATH}" ]] || die "zip not found: ${ZIP_PATH}"
    mapfile -t CASES < <(unzip -Z1 "${ZIP_PATH}" | grep '^test/mdin_.*\.txt$' | sort)
    [[ "${#CASES[@]}" -gt 0 ]] || die "no mdin cases found in ${ZIP_PATH}"

    local entry case_name
    for entry in "${CASES[@]}"; do
        case_name="$(basename "${entry}" .txt)"
        unzip -p "${ZIP_PATH}" "${entry}" > "${OUT_DIR}/cases/${case_name}.orig.mdin"
    done
}

filter_cases_if_needed() {
    [[ -n "${CASES_CSV}" ]] || return

    local raw_name case_name
    declare -A wanted=()
    IFS=',' read -r -a raw_cases <<< "${CASES_CSV}"
    for raw_name in "${raw_cases[@]}"; do
        case_name="${raw_name%.txt}"
        wanted["${case_name}"]=1
    done

    local entry
    local filtered=()
    for entry in "${CASES[@]}"; do
        case_name="$(basename "${entry}" .txt)"
        if [[ -n "${wanted[${case_name}]+x}" ]]; then
            filtered+=("${entry}")
        fi
    done

    CASES=("${filtered[@]}")
    [[ "${#CASES[@]}" -gt 0 ]] || die "CASES_CSV filter removed all cases"
}

rewrite_mdin() {
    local input_mdin="$1"
    local output_mdin="$2"
    local data_root="$3"
    local mdout_path="$4"
    local mdinfo_path="$5"

    awk -v root="${data_root}" \
        -v step_limit="${STEP_LIMIT}" \
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
                if (key=="default_in_file_prefix") {print "default_in_file_prefix = " root "/covid-tip4p/covid-tip4p"; seen_prefix=1; next}
                if (key=="coordinate_in_file") {print "coordinate_in_file = " root "/covid-tip4p/npt_coordinate.txt"; seen_coord=1; next}
                if (key=="velocity_in_file") {print "velocity_in_file = " root "/covid-tip4p/npt_velocity.txt"; seen_vel=1; next}
                if (key=="mdout") {print "mdout = " mdout; seen_mdout=1; next}
                if (key=="mdinfo") {print "mdinfo = " mdinfo; seen_mdinfo=1; next}
            }
            print line
        }
        END{
            if(!seen_step) print "step_limit = " step_limit
            if(!seen_wi) print "write_information_interval = " write_interval
            if(!seen_prefix) print "default_in_file_prefix = " root "/covid-tip4p/covid-tip4p"
            if(!seen_coord) print "coordinate_in_file = " root "/covid-tip4p/npt_coordinate.txt"
            if(!seen_vel) print "velocity_in_file = " root "/covid-tip4p/npt_velocity.txt"
            if(!seen_mdout) print "mdout = " mdout
            if(!seen_mdinfo) print "mdinfo = " mdinfo
        }
    ' "${input_mdin}" > "${output_mdin}"
}

parse_mdout_to_metrics() {
    local mdout_file="$1"
    local metrics_csv="$2"

    echo "Step,Time,Temperature,Potential,LJ,PME,Nb14_LJ,Nb14_EE,Bond,Angle,Dihedral" > "${metrics_csv}"

    if [[ ! -s "${mdout_file}" ]]; then
        return
    fi

    local header header_lc
    header="$(head -n 1 "${mdout_file}" | tr -d '\r')"
    header_lc="$(echo "${header}" | tr 'A-Z' 'a-z')"

    if [[ "${header_lc}" == step,* ]]; then
        awk -F',' '
            function idx(name, arr, n, i) {
                for (i = 1; i <= n; ++i) {
                    if (tolower(arr[i]) == tolower(name)) return i
                }
                return 0
            }
            NR == 1 {
                n = split($0, h, ",")
                for (i = 1; i <= n; ++i) {
                    gsub(/^[ \t]+|[ \t]+$/, "", h[i])
                }
                i_step=idx("step", h, n)
                i_time=idx("time", h, n)
                i_temp=idx("temperature", h, n)
                i_pot=idx("potential", h, n)
                i_lj=idx("lj", h, n)
                i_pme=idx("pme", h, n)
                i_nb14_lj=idx("nb14_lj", h, n)
                i_nb14_ee=idx("nb14_ee", h, n)
                i_bond=idx("bond", h, n)
                i_angle=idx("angle", h, n)
                i_dih=idx("dihedral", h, n)
                next
            }
            NR > 1 {
                if (i_step == 0 || i_time == 0 || i_temp == 0 || i_pot == 0 || i_lj == 0 ||
                    i_pme == 0 || i_nb14_lj == 0 || i_nb14_ee == 0 || i_bond == 0 || i_angle == 0 || i_dih == 0) {
                    next
                }
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", $i_step,$i_time,$i_temp,$i_pot,$i_lj,$i_pme,$i_nb14_lj,$i_nb14_ee,$i_bond,$i_angle,$i_dih
            }
        ' "${mdout_file}" >> "${metrics_csv}"
    else
        awk '
            BEGIN { OFS="," }
            NR <= 1 { next }
            {
                line=$0
                gsub(/\r/, "", line)
                gsub(/^ +| +$/, "", line)
                gsub(/ +/, " ", line)
                n=split(line, a, " ")
                if (n >= 11 && a[1] ~ /^[0-9]+$/) {
                    print a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11]
                }
            }
        ' "${mdout_file}" >> "${metrics_csv}"
    fi
}

scan_nonfinite_in_log() {
    local log_file="$1"
    if rg -i -n -m 1 'non[- ]finite|(^|[^A-Za-z0-9_])(nan|inf)([^A-Za-z0-9_]|$)|spongeerror|SPONGE_ERROR' "${log_file}" >/dev/null 2>&1; then
        echo "1"
    else
        echo "0"
    fi
}

last_step_from_metrics() {
    local metrics_csv="$1"
    if [[ ! -s "${metrics_csv}" ]]; then
        echo "N/A"
        return
    fi
    awk -F',' 'NR>1{last=$1} END{if(last=="") print "N/A"; else print last}' "${metrics_csv}"
}

append_run_index_row() {
    local row="$1"
    (
        flock -x 200
        echo "${row}" >> "${OUT_DIR}/run_index.csv"
    ) 200>"${OUT_DIR}/run_index.lock"
}

run_single() {
    local backend="$1"
    local case_name="$2"
    local run_id="$3"

    local tag="${case_name}.${backend}.run${run_id}"
    local mdin_src="${OUT_DIR}/cases/${case_name}.orig.mdin"
    local mdin_path="${OUT_DIR}/mdin/${tag}.mdin"
    local log_path="${OUT_DIR}/logs/${tag}.log"
    local mdout_path="${OUT_DIR}/mdout/${tag}.mdout.txt"
    local mdinfo_path="${OUT_DIR}/mdinfo/${tag}.mdinfo.txt"
    local metrics_path="${OUT_DIR}/metrics/${tag}.metrics.csv"

    local data_root
    data_root="$(backend_source_dir "${backend}")"

    rewrite_mdin "${mdin_src}" "${mdin_path}" "${data_root}" "${mdout_path}" "${mdinfo_path}"

    normalize_line_endings_if_needed "${backend}"

    local bin
    bin="$(backend_binary "${backend}")"

    local t0 t1 elapsed exit_code timeout_s
    t0="$(date +%s)"
    if [[ "${backend}" == "gpu_ref" ]]; then
        timeout_s="${GPU_TIMEOUT_SECONDS}"
    else
        timeout_s="${CPU_TIMEOUT_SECONDS}"
    fi

    log "run ${tag}"
    set +e
    if [[ "${backend}" == "local_cpu" ]]; then
        SPONGE_CPU_OMP_LAUNCH=0 timeout "${timeout_s}s" "${bin}" -mdin "${mdin_path}" > "${log_path}" 2>&1
        exit_code="$?"
    else
        timeout "${timeout_s}s" "${bin}" -mdin "${mdin_path}" > "${log_path}" 2>&1
        exit_code="$?"
    fi
    set -e
    t1="$(date +%s)"
    elapsed="$((t1 - t0))"

    parse_mdout_to_metrics "${mdout_path}" "${metrics_path}"

    local nonfinite_flag last_step
    nonfinite_flag="$(scan_nonfinite_in_log "${log_path}")"
    last_step="$(last_step_from_metrics "${metrics_path}")"

    append_run_index_row "${backend},${case_name},${run_id},${exit_code},${elapsed},${last_step},${nonfinite_flag},${log_path},${metrics_path}"
}

run_backends_for_case() {
    local case_name="$1"
    local run_id="$2"
    local backend running=0 failed=0

    for backend in "${BACKENDS[@]}"; do
        if [[ "${PARALLEL_BACKENDS}" -eq 1 ]]; then
            run_single "${backend}" "${case_name}" "${run_id}"
            continue
        fi

        run_single "${backend}" "${case_name}" "${run_id}" &
        running=$((running + 1))
        if [[ "${running}" -ge "${PARALLEL_BACKENDS}" ]]; then
            if ! wait -n; then
                failed=1
            fi
            running=$((running - 1))
        fi
    done

    while [[ "${running}" -gt 0 ]]; do
        if ! wait -n; then
            failed=1
        fi
        running=$((running - 1))
    done

    [[ "${failed}" -eq 0 ]] || die "one or more backend runs failed for case=${case_name}, run_id=${run_id}"
}

generate_relerr_for_backend_case() {
    local backend="$1"
    local case_name="$2"

    local gpu_metrics="${OUT_DIR}/metrics/${case_name}.gpu_ref.run1.metrics.csv"
    local cpu_metrics="${OUT_DIR}/metrics/${case_name}.${backend}.run1.metrics.csv"
    local relerr_csv="${OUT_DIR}/errors/${case_name}.${backend}.relerr.csv"

    echo "Step,Time,Temperature,Potential,LJ,PME,Nb14_LJ,Nb14_EE,Bond,Angle,Dihedral" > "${relerr_csv}"

    awk -F',' -v step_start="${ERR_STEP_START}" -v step_end="${ERR_STEP_END}" -v step_inc="${ERR_STEP_INTERVAL}" '
        function absf(x){return x<0?-x:x}
        function isnum(x){return x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/}
        function rel(cpu,gpu){d=absf(cpu-gpu);den=absf(gpu);if(den<1e-12) den=1.0; return 100.0*d/den}
        NR==FNR {
            if (NR == 1) next
            g[$1]=$0
            next
        }
        NR>1 {
            c[$1]=$0
        }
        END {
            for (step=step_start; step<=step_end; step+=step_inc) {
                skey = sprintf("%d", step)
                if (!(skey in c) || !(skey in g)) {
                    printf "%d,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n", step
                    continue
                }
                split(c[skey], C, ",")
                split(g[skey], G, ",")
                out[1] = isnum(C[2]) && isnum(G[2]) ? sprintf("%.6f", rel(C[2], G[2])) : "N/A"
                out[2] = isnum(C[3]) && isnum(G[3]) ? sprintf("%.6f", rel(C[3], G[3])) : "N/A"
                out[3] = isnum(C[4]) && isnum(G[4]) ? sprintf("%.6f", rel(C[4], G[4])) : "N/A"
                out[4] = isnum(C[5]) && isnum(G[5]) ? sprintf("%.6f", rel(C[5], G[5])) : "N/A"
                out[5] = isnum(C[6]) && isnum(G[6]) ? sprintf("%.6f", rel(C[6], G[6])) : "N/A"
                out[6] = isnum(C[7]) && isnum(G[7]) ? sprintf("%.6f", rel(C[7], G[7])) : "N/A"
                out[7] = isnum(C[8]) && isnum(G[8]) ? sprintf("%.6f", rel(C[8], G[8])) : "N/A"
                out[8] = isnum(C[9]) && isnum(G[9]) ? sprintf("%.6f", rel(C[9], G[9])) : "N/A"
                out[9] = isnum(C[10]) && isnum(G[10]) ? sprintf("%.6f", rel(C[10], G[10])) : "N/A"
                out[10] = isnum(C[11]) && isnum(G[11]) ? sprintf("%.6f", rel(C[11], G[11])) : "N/A"
                printf "%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", step,
                    out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10]
            }
        }
    ' "${gpu_metrics}" "${cpu_metrics}" >> "${relerr_csv}"

    awk -F',' '
        function isnum(x){return x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/}
        NR==1 {next}
        {
            for (i=2; i<=11; ++i) {
                if (isnum($i)) {
                    sum += $i
                    cnt += 1
                    if ($i > maxv) maxv = $i
                }
            }
        }
        END {
            mean = (cnt>0 ? sum/cnt : -1)
            if (cnt == 0) {
                print "N/A,N/A"
            } else {
                printf "%.6f,%.6f\n", mean, maxv
            }
        }
    ' "${relerr_csv}" > "${relerr_csv}.summary"
}

generate_all_relative_errors() {
    echo "backend,case,mean_rel_err_pct,max_rel_err_pct" > "${OUT_DIR}/relerr_summary.csv"
    local backend case_name stats mean maxv
    for backend in "${CPU_BACKENDS[@]}"; do
        for case_name in "${CASES[@]}"; do
            case_name="$(basename "${case_name}" .txt)"
            generate_relerr_for_backend_case "${backend}" "${case_name}"
            stats="${OUT_DIR}/errors/${case_name}.${backend}.relerr.csv.summary"
            mean="$(cut -d',' -f1 "${stats}")"
            maxv="$(cut -d',' -f2 "${stats}")"
            echo "${backend},${case_name},${mean},${maxv}" >> "${OUT_DIR}/relerr_summary.csv"
        done
    done
}

run_matrix() {
    : > "${OUT_DIR}/run_index.csv"
    echo "backend,case,run_id,exit_code,elapsed_s,last_step,nonfinite_flag,log_path,metrics_path" > "${OUT_DIR}/run_index.csv"
    : > "${OUT_DIR}/run_index.lock"

    local entry case_name
    for entry in "${CASES[@]}"; do
        case_name="$(basename "${entry}" .txt)"
        run_backends_for_case "${case_name}" "1"
    done

    if [[ "${SKIP_REPEAT}" != "1" && -n "${REPEAT_CASE}" ]]; then
        local found=0
        for entry in "${CASES[@]}"; do
            case_name="$(basename "${entry}" .txt)"
            if [[ "${case_name}" == "${REPEAT_CASE}" ]]; then
                found=1
                break
            fi
        done
        if [[ "${found}" -eq 1 ]]; then
            run_backends_for_case "${REPEAT_CASE}" "2"
        else
            log "skip repeat run: case ${REPEAT_CASE} not in selected case list"
        fi
    fi
}

generate_backend_case_matrix() {
    local out_csv="${OUT_DIR}/backend_case_laststep.csv"
    echo "backend,case,run_id,step,time,temp,potential,lj,pme,nb14_lj,nb14_ee,bond,angle,dihedral" > "${out_csv}"

    local backend case_name run_id metrics_file last_line
    while IFS=',' read -r backend case_name run_id _ _ _ _ _ metrics_file; do
        [[ "${backend}" == "backend" ]] && continue
        if [[ -s "${metrics_file}" ]]; then
            last_line="$(tail -n 1 "${metrics_file}")"
            echo "${backend},${case_name},${run_id},${last_line}" >> "${out_csv}"
        else
            echo "${backend},${case_name},${run_id},N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A" >> "${out_csv}"
        fi
    done < "${OUT_DIR}/run_index.csv"
}

main() {
    ensure_tools
    [[ -x "${GPU_BIN}" ]] || die "GPU binary not executable: ${GPU_BIN}"

    configure_backends
    prepare_directories
    copy_sources_to_staging

    local backend
    for backend in "${CPU_BACKENDS[@]}"; do
        build_backend "${backend}"
    done

    extract_cases_from_zip
    filter_cases_if_needed
    run_matrix
    generate_backend_case_matrix
    generate_all_relative_errors

    log "matrix run complete"
    log "run index: ${OUT_DIR}/run_index.csv"
    log "relative error summary: ${OUT_DIR}/relerr_summary.csv"
}

main "$@"
