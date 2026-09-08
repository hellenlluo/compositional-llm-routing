#!/usr/bin/env bash
# Submit one SLURM job per (model, family) pair. Each job loops over all
# categories within that family sequentially, skipping any already done.
#
# Usage:
#   bash embedllm_eval/submit_per_model_jobs.sh                              # all models
#   bash embedllm_eval/submit_per_model_jobs.sh --dry-run                    # preview
#   bash embedllm_eval/submit_per_model_jobs.sh --model mathstral-7b-v0.1   # one model

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
mkdir -p "$SLURM_LOGS"

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
PARTITION="${PARTITION:-all}"
ACCOUNT="${ACCOUNT:-allcs}"
GRES="${GRES:-gpu:a6000:2}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-128G}"
CPUS="${CPUS:-8}"
BACKEND="${BACKEND:-auto}"
BATCH_SIZE="${BATCH_SIZE:-3}"
FILTER_MODEL=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --account)    ACCOUNT="$2"; shift 2 ;;
        --gres)       GRES="$2"; shift 2 ;;
        --time)       TIME="$2"; shift 2 ;;
        --mem)        MEM="$2"; shift 2 ;;
        --backend)    BACKEND="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --model)      FILTER_MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Models list
# --------------------------------------------------------------------------
ALL_MODELS=(
    "qwen3-4b-thinking-2507"
    "deepseek-r1-distill-llama-8b"
    "medgemma-4b-it"
    "llama-3.1-8b-instruct"
    "qwen3-30b-a3b"
    "phi-4-mini-instruct"
    "mistral-7b-instruct-v0.3"
    "qwen1.5-0.5b-chat"
    "mathstral-7b-v0.1"
    "llama-3.1-nemotron-nano-8b-v1"
)

if [[ -n "$FILTER_MODEL" ]]; then
    ALL_MODELS=("$FILTER_MODEL")
fi

# --------------------------------------------------------------------------
# Family → category mapping (derived from prompts_subset.jsonl)
# --------------------------------------------------------------------------
declare -A FAMILY_CATS

FAMILY_CATS[ASDiv]="asdiv"
FAMILY_CATS[GPQA]="gpqa_diamond_cot_n_shot gpqa_diamond_cot_zeroshot gpqa_diamond_generative_n_shot gpqa_diamond_n_shot gpqa_diamond_zeroshot gpqa_extended_cot_n_shot gpqa_extended_cot_zeroshot gpqa_extended_generative_n_shot gpqa_extended_n_shot gpqa_extended_zeroshot gpqa_main_cot_n_shot gpqa_main_cot_zeroshot gpqa_main_generative_n_shot gpqa_main_n_shot gpqa_main_zeroshot"
FAMILY_CATS[GSM8K]="gsm8k"
FAMILY_CATS[LogiQA]="logiqa"
FAMILY_CATS[MMLU]="mmlu_abstract_algebra mmlu_anatomy mmlu_astronomy mmlu_business_ethics mmlu_clinical_knowledge mmlu_college_biology mmlu_college_chemistry mmlu_college_computer_science mmlu_college_mathematics mmlu_college_medicine mmlu_college_physics mmlu_computer_security mmlu_conceptual_physics mmlu_econometrics mmlu_electrical_engineering mmlu_elementary_mathematics mmlu_formal_logic mmlu_global_facts mmlu_high_school_biology mmlu_high_school_chemistry mmlu_high_school_computer_science mmlu_high_school_european_history mmlu_high_school_geography mmlu_high_school_government_and_politics mmlu_high_school_macroeconomics mmlu_high_school_mathematics mmlu_high_school_microeconomics mmlu_high_school_physics mmlu_high_school_psychology mmlu_high_school_statistics mmlu_high_school_us_history mmlu_high_school_world_history mmlu_human_aging mmlu_human_sexuality mmlu_international_law mmlu_jurisprudence mmlu_logical_fallacies mmlu_machine_learning mmlu_management mmlu_marketing mmlu_medical_genetics mmlu_miscellaneous mmlu_moral_disputes mmlu_moral_scenarios mmlu_nutrition mmlu_philosophy mmlu_prehistory mmlu_professional_accounting mmlu_professional_law mmlu_professional_medicine mmlu_professional_psychology mmlu_public_relations mmlu_security_studies mmlu_sociology mmlu_us_foreign_policy mmlu_virology mmlu_world_religions"
FAMILY_CATS[MathQA]="mathqa"
FAMILY_CATS[MedMCQA]="medmcqa"
FAMILY_CATS[PIQA]="piqa"
FAMILY_CATS[SocialIQA]="social_iqa"
FAMILY_CATS[TruthfulQA]="Conspiracies Distraction Economics Health Misquotations advertising confusion_other confusion_people confusion_places conspiracies distraction economics education fiction health history indexical_error_location language law misconceptions misinformation myths_and_fairytales nutrition paranormal psychology religion sociology stereotypes subjective superstitions weather"

FAMILIES=(ASDiv GPQA GSM8K LogiQA MMLU MathQA MedMCQA PIQA SocialIQA TruthfulQA)

N_MODELS=${#ALL_MODELS[@]}
N_FAMILIES=${#FAMILIES[@]}
echo "=== EmbedLLM Evaluation — submitting ${N_MODELS} models × ${N_FAMILIES} families = $((N_MODELS * N_FAMILIES)) jobs ==="
echo "  partition=${PARTITION}  account=${ACCOUNT}  gres=${GRES}  time=${TIME}  mem=${MEM}"
echo "  backend=${BACKEND}  batch_size=${BATCH_SIZE}"
echo ""

SUBMITTED=0

for MODEL_ID in "${ALL_MODELS[@]}"; do
    for FAMILY in "${FAMILIES[@]}"; do
        CATS="${FAMILY_CATS[$FAMILY]}"
        N_CATS=$(echo "$CATS" | wc -w | tr -d ' ')

        # Check if all categories for this family are already done
        ALL_DONE=true
        for CATEGORY in $CATS; do
            RESULT_FILE="$REPO_ROOT/embedllm_eval/results/$MODEL_ID/$CATEGORY.json"
            if [[ ! -f "$RESULT_FILE" || ! -s "$RESULT_FILE" ]]; then
                ALL_DONE=false
                break
            fi
        done
        if $ALL_DONE; then
            echo "[SKIP] $MODEL_ID / $FAMILY — all $N_CATS categories done"
            continue
        fi

        JOB_NAME="emb-${MODEL_ID:0:12}-${FAMILY:0:10}"
        LOG_FILE="$SLURM_LOGS/${MODEL_ID}__${FAMILY}_%j.out"

        # Build inline loop over categories in this family
        INNER_LOOP="cd $REPO_ROOT"
        for CATEGORY in $CATS; do
            RESULT_FILE="$REPO_ROOT/embedllm_eval/results/$MODEL_ID/$CATEGORY.json"
            INNER_LOOP+=" && { [[ -f '$RESULT_FILE' && -s '$RESULT_FILE' ]] && echo '[SKIP] $CATEGORY — already done' || \
                HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
                /n/fs/scratch/dl3533/conda_envs/MARIO/bin/python embedllm_eval/run_one_job.py \
                --model-id $MODEL_ID \
                --category $CATEGORY \
                --backend $BACKEND \
                --batch-size $BATCH_SIZE; }"
        done

        SBATCH_CMD="sbatch \
            --job-name=$JOB_NAME \
            --partition=$PARTITION \
            --account=$ACCOUNT \
            --gres=$GRES \
            --time=$TIME \
            --mem=$MEM \
            --cpus-per-task=$CPUS \
            --output=$LOG_FILE \
            --error=$LOG_FILE \
            --wrap=\"$INNER_LOOP\""

        if $DRY_RUN; then
            echo "[DRY-RUN] $MODEL_ID / $FAMILY ($N_CATS categories)"
        else
            eval "$SBATCH_CMD"
            SUBMITTED=$(( SUBMITTED + 1 ))
        fi
    done
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete."
else
    echo "Submitted $SUBMITTED jobs."
    echo "Monitor with: squeue -u \$USER"
    echo "After all jobs finish, run: python embedllm_eval/stitch_results.py"
fi
