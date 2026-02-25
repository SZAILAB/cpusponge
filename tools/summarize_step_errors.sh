#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/reports/step1000}"
GPU_BIN="${GPU_BIN:-/home/wuping/SPONGE/SPONGE}"

RUN_INDEX="${OUT_DIR}/run_index.csv"
RELERR_SUMMARY="${OUT_DIR}/relerr_summary.csv"
SCORES_CSV="${OUT_DIR}/backend_scores.csv"
MAINT_CSV="${OUT_DIR}/maintainability.csv"
FINAL_REPORT="${OUT_DIR}/final_report.md"
PASS_STEP_THRESHOLD="${PASS_STEP_THRESHOLD:-1000}"

CANDIDATES=(local_cpu)
BACKENDS=(gpu_ref local_cpu)
REQUIRED_MODULES=(
    Lennard_Jones_force MD_core No_PBC PME_force SITS angle barostats bias
    bond cmap collective_variable constrain dihedral nb14 neighbor_list
    plugin restrain thermostats virtual_atoms wall
)

is_num() {
    [[ "${1}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]
}

clamp_0_100() {
    awk -v x="$1" 'BEGIN{if (x<0) x=0; if (x>100) x=100; printf "%.4f", x}'
}

calc_repeat_drift_pct() {
    local metrics_run1="$1"
    local metrics_run2="$2"
    if [[ ! -s "${metrics_run1}" || ! -s "${metrics_run2}" ]]; then
        echo "N/A"
        return
    fi

    awk -F',' '
        function absf(x){return x<0?-x:x}
        function isnum(x){return x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/}
        END{
            # no-op placeholder
        }
    ' /dev/null >/dev/null

    local line1 line2
    line1="$(tail -n 1 "${metrics_run1}")"
    line2="$(tail -n 1 "${metrics_run2}")"

    awk -F',' -v a="${line1}" -v b="${line2}" '
        function absf(x){return x<0?-x:x}
        function isnum(x){return x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/}
        BEGIN {
            split(a, A, ",")
            split(b, B, ",")
            sum=0
            cnt=0
            for (i=2; i<=11; ++i) {
                if (isnum(A[i]) && isnum(B[i])) {
                    d=absf(A[i]-B[i])
                    den=absf(A[i])
                    if (den < 1e-12) den = 1.0
                    sum += 100.0*d/den
                    cnt += 1
                }
            }
            if (cnt==0) print "N/A";
            else printf "%.6f", sum/cnt
        }
    '
}

compute_maintainability() {
    echo "backend,module_completeness_pct,artifact_count,crlf_count,maintainability_score" > "${MAINT_CSV}"

    local backend present total completeness artifacts crlf score
    total="${#REQUIRED_MODULES[@]}"

    for backend in "${CANDIDATES[@]}"; do
        present=0
        local module
        for module in "${REQUIRED_MODULES[@]}"; do
            if [[ -d "${ROOT_DIR}/${module}" ]]; then
                present=$((present + 1))
            fi
        done

        completeness="$(awk -v p="${present}" -v t="${total}" 'BEGIN{printf "%.4f", (p*100.0)/t}')"

        artifacts="$(find "${ROOT_DIR}" -maxdepth 2 \( -name '*.o' -o -name 'SPONGE_CPU' -o -name 'SPONGE_CPU_TI' -o -name 'sponge_cpu_mainpath' \) 2>/dev/null | wc -l | awk '{print $1}')"

        if [[ -d "${ROOT_DIR}/covid-tip4p" ]]; then
            crlf="$( { rg -U -n '\r$' "${ROOT_DIR}/covid-tip4p" 2>/dev/null || true; } | wc -l | awk '{print $1}')"
        else
            crlf=0
        fi

        score="${completeness}"
        if [[ "${artifacts}" -gt 0 ]]; then
            score="$(awk -v s="${score}" 'BEGIN{printf "%.4f", s-10.0}')"
        fi
        if [[ "${crlf}" -gt 0 ]]; then
            score="$(awk -v s="${score}" 'BEGIN{printf "%.4f", s-20.0}')"
        fi
        if [[ "${backend}" == "local_cpu" ]]; then
            score="$(awk -v s="${score}" 'BEGIN{printf "%.4f", s+5.0}')"
        fi
        score="$(clamp_0_100 "${score}")"

        echo "${backend},${completeness},${artifacts},${crlf},${score}" >> "${MAINT_CSV}"
    done
}

load_min_refs() {
    local min_acc=""
    local min_speed=""

    local backend
    for backend in "${CANDIDATES[@]}"; do
        local acc
        acc="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b && $3!="N/A"{sum+=$3;cnt+=1} END{if(cnt==0) print "N/A"; else printf "%.6f", sum/cnt}' "${RELERR_SUMMARY}")"
        local speed
        speed="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b && $3==1{sum+=$5;cnt+=1} END{if(cnt==0) print "N/A"; else printf "%.6f", sum/cnt}' "${RUN_INDEX}")"

        if is_num "${acc}"; then
            if [[ -z "${min_acc}" ]] || awk -v x="${acc}" -v y="${min_acc}" 'BEGIN{exit !(x<y)}'; then
                min_acc="${acc}"
            fi
        fi
        if is_num "${speed}"; then
            if [[ -z "${min_speed}" ]] || awk -v x="${speed}" -v y="${min_speed}" 'BEGIN{exit !(x<y)}'; then
                min_speed="${speed}"
            fi
        fi
    done

    [[ -n "${min_acc}" ]] || min_acc="1"
    [[ -n "${min_speed}" ]] || min_speed="1"

    echo "${min_acc},${min_speed}"
}

calc_scores() {
    [[ -f "${RUN_INDEX}" ]] || { echo "missing run index: ${RUN_INDEX}" >&2; exit 1; }
    [[ -f "${RELERR_SUMMARY}" ]] || { echo "missing relerr summary: ${RELERR_SUMMARY}" >&2; exit 1; }
    [[ -f "${MAINT_CSV}" ]] || { echo "missing maintainability csv: ${MAINT_CSV}" >&2; exit 1; }

    local refs min_acc min_speed
    refs="$(load_min_refs)"
    min_acc="${refs%,*}"
    min_speed="${refs#*,}"

    echo "backend,accuracy_mean_err_pct,accuracy_score,stability_score,speed_avg_elapsed_s,speed_score,maintainability_score,overall_score,pass_main,nonfinite_main,repeat_drift_pct,status" > "${SCORES_CSV}"

    local backend
    for backend in "${BACKENDS[@]}"; do
        if [[ "${backend}" == "gpu_ref" ]]; then
            echo "gpu_ref,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,REFERENCE" >> "${SCORES_CSV}"
            continue
        fi

        local acc_mean
        acc_mean="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b && $3!="N/A"{sum+=$3;cnt+=1} END{if(cnt==0) print "N/A"; else printf "%.6f", sum/cnt}' "${RELERR_SUMMARY}")"

        local pass_main nonfinite_main avg_elapsed
        pass_main="$(awk -F',' -v b="${backend}" -v step_min="${PASS_STEP_THRESHOLD}" 'NR>1 && $1==b && $3==1 && $4==0 && $6+0>=step_min && $7==0 {cnt+=1} END{print cnt+0}' "${RUN_INDEX}")"
        nonfinite_main="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b && $3==1 {sum+=$7} END{print sum+0}' "${RUN_INDEX}")"
        avg_elapsed="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b && $3==1 {sum+=$5;cnt+=1} END{if(cnt==0) print "N/A"; else printf "%.6f", sum/cnt}' "${RUN_INDEX}")"

        local repeat_m1 repeat_m2 repeat_drift
        repeat_m1="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b && $2=="mdin_npt_mcbaro" && $3==1 {print $9}' "${RUN_INDEX}" | tail -n 1)"
        repeat_m2="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b && $2=="mdin_npt_mcbaro" && $3==2 {print $9}' "${RUN_INDEX}" | tail -n 1)"
        repeat_drift="$(calc_repeat_drift_pct "${repeat_m1}" "${repeat_m2}")"

        local maint
        maint="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b {print $5}' "${MAINT_CSV}")"
        [[ -n "${maint}" ]] || maint="0"

        local acc_score speed_score stability_score overall_score status
        if is_num "${acc_mean}"; then
            acc_score="$(awk -v min="${min_acc}" -v x="${acc_mean}" 'BEGIN{if(x<=0){print "100.0000"}else{printf "%.4f", 100.0*min/x}}')"
            acc_score="$(clamp_0_100 "${acc_score}")"
        else
            acc_score="0.0000"
        fi

        if is_num "${avg_elapsed}"; then
            speed_score="$(awk -v min="${min_speed}" -v x="${avg_elapsed}" 'BEGIN{if(x<=0){print "0.0000"}else{printf "%.4f", 100.0*min/x}}')"
            speed_score="$(clamp_0_100 "${speed_score}")"
        else
            speed_score="0.0000"
        fi

        stability_score="$(awk -v pass="${pass_main}" -v nonf="${nonfinite_main}" -v drift="${repeat_drift}" 'BEGIN{
            s = 100.0 * pass / 7.0;
            s -= 20.0 * nonf;
            if (drift != "N/A") {
                d = drift + 0.0;
                pen = d * 2.0;
                if (pen > 30.0) pen = 30.0;
                s -= pen;
            } else {
                s -= 20.0;
            }
            if (s < 0.0) s = 0.0;
            if (s > 100.0) s = 100.0;
            printf "%.4f", s;
        }')"

        status="PASS"
        if [[ "${pass_main}" -lt 7 || "${nonfinite_main}" -gt 0 ]]; then
            status="FAIL_GATE"
        fi
        if ! is_num "${acc_mean}"; then
            status="FAIL_GATE"
        fi

        overall_score="$(awk -v a="${acc_score}" -v st="${stability_score}" -v sp="${speed_score}" -v m="${maint}" 'BEGIN{printf "%.4f", 0.70*a + 0.15*st + 0.10*sp + 0.05*m}')"
        if [[ "${status}" == "FAIL_GATE" ]]; then
            overall_score="0.0000"
        fi

        echo "${backend},${acc_mean},${acc_score},${stability_score},${avg_elapsed},${speed_score},${maint},${overall_score},${pass_main},${nonfinite_main},${repeat_drift},${status}" >> "${SCORES_CSV}"
    done
}

build_final_report() {
    local best_backend
    best_backend="$(awk -F',' 'NR>1 && $12=="PASS" {if($8+0 > best){best=$8+0; b=$1}} END{if(b=="") print "N/A"; else print b}' "${SCORES_CSV}")"

    {
        echo "# CPU SPONGE Step1000 全量评测报告"
        echo
        echo "- 时间: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
        echo "- 输入: ${ROOT_DIR}/test.zip"
        echo "- GPU参考: ${GPU_BIN}"
        echo "- 比较步点: 100,200,...,1000"
        echo
        echo "## 评分权重"
        echo "- Accuracy 70%"
        echo "- Stability 15%"
        echo "- Speed 10%"
        echo "- Maintainability 5%"
        echo
        echo "## 最终推荐"
        echo "- Best Backend: ${best_backend}"
        echo
        echo "## 版本得分"
        echo "| Backend | Acc Mean Err % | Acc Score | Stability | Avg Elapsed(s) | Speed | Maintainability | Overall | PassMain | NonFinite | Repeat Drift % | Status |"
        echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        awk -F',' 'NR>1{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12}' "${SCORES_CSV}"
        echo
        echo "## 可维护性检查"
        echo "| Backend | Module Completeness % | Artifact Count | CRLF Count | Maintainability Score |"
        echo "|---|---:|---:|---:|---:|"
        awk -F',' 'NR>1{printf "| %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5}' "${MAINT_CSV}"
        echo
        echo "## 每版本优缺点"

        local backend
        for backend in "${CANDIDATES[@]}"; do
            local row
            row="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b {print $0}' "${SCORES_CSV}")"
            local acc speed st maint status
            acc="$(echo "${row}" | awk -F',' '{print $3}')"
            st="$(echo "${row}" | awk -F',' '{print $4}')"
            speed="$(echo "${row}" | awk -F',' '{print $6}')"
            maint="$(echo "${row}" | awk -F',' '{print $7}')"
            status="$(echo "${row}" | awk -F',' '{print $12}')"

            echo "### ${backend}"
            echo "- Pros:"
            if is_num "${acc}" && awk -v x="${acc}" 'BEGIN{exit !(x>=90)}'; then echo "  - 精度分高，数值贴近GPU。"; fi
            if is_num "${st}" && awk -v x="${st}" 'BEGIN{exit !(x>=90)}'; then echo "  - 稳定性高，主case通过率好。"; fi
            if is_num "${speed}" && awk -v x="${speed}" 'BEGIN{exit !(x>=70)}'; then echo "  - 运行速度在CPU候选中较快。"; fi
            if is_num "${maint}" && awk -v x="${maint}" 'BEGIN{exit !(x>=80)}'; then echo "  - 模块覆盖完整，可维护性较好。"; fi
            if ! (is_num "${acc}" && awk -v x="${acc}" 'BEGIN{exit !(x>=90)}') && \
               ! (is_num "${st}" && awk -v x="${st}" 'BEGIN{exit !(x>=90)}') && \
               ! (is_num "${speed}" && awk -v x="${speed}" 'BEGIN{exit !(x>=70)}') && \
               ! (is_num "${maint}" && awk -v x="${maint}" 'BEGIN{exit !(x>=80)}'); then
                echo "  - 无显著优势项。"
            fi
            echo "- Cons:"
            if [[ "${status}" != "PASS" ]]; then echo "  - 未通过7/7与non-finite门槛。"; fi
            if is_num "${acc}" && awk -v x="${acc}" 'BEGIN{exit !(x<80)}'; then echo "  - 精度分偏低，误差偏大。"; fi
            if is_num "${speed}" && awk -v x="${speed}" 'BEGIN{exit !(x<40)}'; then echo "  - 运行速度较慢。"; fi
            if is_num "${maint}" && awk -v x="${maint}" 'BEGIN{exit !(x<70)}'; then echo "  - 模块覆盖或工程洁净度不足。"; fi
            local crlf
            crlf="$(awk -F',' -v b="${backend}" 'NR>1 && $1==b {print $4}' "${MAINT_CSV}")"
            if [[ -n "${crlf}" && "${crlf}" -gt 0 ]]; then echo "  - 存在CRLF输入文件，跨平台运行需要归一化。"; fi
            echo
        done

        echo "## 关键产物"
        echo "- ${OUT_DIR}/run_index.csv"
        echo "- ${OUT_DIR}/backend_case_laststep.csv"
        echo "- ${OUT_DIR}/relerr_summary.csv"
        echo "- ${OUT_DIR}/errors/*.relerr.csv"
        echo "- ${OUT_DIR}/backend_scores.csv"
        echo "- ${OUT_DIR}/maintainability.csv"
    } > "${FINAL_REPORT}"
}

main() {
    [[ -f "${RUN_INDEX}" ]] || { echo "missing ${RUN_INDEX}" >&2; exit 1; }
    [[ -f "${RELERR_SUMMARY}" ]] || { echo "missing ${RELERR_SUMMARY}" >&2; exit 1; }

    compute_maintainability
    calc_scores
    build_final_report

    echo "summary generated: ${FINAL_REPORT}"
    echo "scores csv: ${SCORES_CSV}"
}

main "$@"
