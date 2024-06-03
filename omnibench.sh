#!/bin/bash


############################################################
# USAGE FUNCTIONS                                          #
############################################################

print_common_usage() {
    cat << EOF
== OmniBench

Usage:
  $(basename $0) <command> [options] <arguments>

Commands:
  check      Run a correctness check on a kernel with specific optimization and problem dimensions.
  measure    Perform benchmarking on a kernel with specific optimization, benchmark type, and problem dimensions.
  export     Export GPU information and CSV data to a markdown file.

For detailed help on each command, use:
  $(basename $0) <command> --help

EOF
}

usage_check() {
    cat << EOF
== OmniBench: Check Command

Usage:
  $(basename $0) check [options] <kernel> <optimization> <problem_dim>

Options:
  -h, --help         Show this help message and exit
  -v, --verbose      Enable verbose mode

Arguments:
  <kernel>           Kernel to use (matrixMultiply, saxpy, matrixTranspose, matrixVectorMultiply, matrixCopy)
  <optimization>     Optimization type (NOPT, TILE, UNROLL, STRIDE)
  <problem_dim>      Problem dimensions (size for vectors, size1xsize2 for matrices)

EOF
}

usage_measure() {
    cat << EOF
== OmniBench: Measure Command

Usage:
  $(basename $0) measure [options] <kernel> <optimization> <benchmark> <problem_dim>

Options:
  -h, --help         Show this help message and exit
  --rerun            Re-run all runs of the benchmark
  -o, --output <f>   Save the benchmark results in the specified file
  -v, --verbose      Enable verbose mode

Arguments:
  <kernel>           Kernel to use (matrixMultiply, saxpy, matrixTranspose, matrixVectorMultiply, matrixCopy)
  <optimization>     Optimization type (NOPT, TILE, UNROLL, STRIDE)
  <benchmark>        Benchmark type (Block size variation, Grid size variation, LDS size variation, Unroll size variation, Striding kernel with block size variation)
  <problem_dim>      Problem dimensions (size for vectors, sizexsize for matrices)

EOF
}

usage_export() {
    cat << EOF
== OmniBench: Export Command

Usage:
  $(basename $0) export [options]

Options:
  -h, --help         Show this help message and exit
  -o, --output <f>   Save the exported markdown file as the specified file
  -i, --input <f>    Specify the input CSV file to be converted to markdown table

EOF
}

usage() {
    case "$CMD" in
        check) usage_check ;;
        measure) usage_measure ;;
        export) usage_export ;;
        *) print_common_usage ;;
    esac
    exit 1
}


############################################################
# ARGS MANAGER                                             #
############################################################

check_option()
{
    verbose=0
    rerun=0
    input=""
    output=""
    type=""
    block_size=32
    grid_size=""
    nrep=1
    plot=0
    plot_file="$WORKDIR/results/graph_$(date +%F-%T).png"

    TEMP=$(getopt -o $opt_list_short \
                    -l $opt_list \
                    -n $(basename $0) -- "$@")
    if [ $? != 0 ]; then usage ; fi
    eval set -- "$TEMP"
    if [ $? != 0 ]; then usage ; fi

    while true ; do
        case "$1" in
            -h|--help) usage ;;
            -v|--verbose) verbose=1 ; shift ;;
            -o|--output) output=($2) ; shift 2 ;;
            -i|--input) input=($2) ; shift 2 ;;
            -n|--nrep) nrep=($2) ; shift 2 ;;
            -p|--plot) 
                case "$2" in
                    "") plot=1; shift 2 ;;
                    *)  plot=1; plot_file="$2" ; shift 2 ;;
                esac ;;
            -b|--block-size) block_size=($2) ; shift 2 ;;
            -g|--grid-size) grid_size=($2) ; shift 2 ;;
            --rerun) rerun=1 ; shift ;;
            --) shift ; break ;;
            *) echo "No option $1."; usage ;;
        esac
    done
    
     ARGS=$@
}

check_args()
{
    if [ "$CMD" == "check" ]; then
        if [ $# -ne 3 ]; then
            echo "Need arguments."
            usage
        fi
        PB_SIZE=$3

    elif [ "$CMD" == "measure" ]; then
        if [ $# -ne 4 ]; then
            echo "Need arguments."
            usage
        fi
        BENCH=$3
        PB_SIZE=$4
    fi 

    KERNEL=$1
    OPT=$2
    BIN_PATH=$WORKDIR/benchmark/$KERNEL/build/bin
    if [ "$grid_size" == "" ]; then
       grid_size=$((($PB_SIZE + $block_size - 1) / $block_size))
    fi
    DIM="ONE_DIM"
    start=8
    end=1024
    step=8
    if [[ "$KERNEL" -eq "matrixMultiply" && "$OPT" -ne "LINEAR" ]]; then
        DIM="TWO_DIM"
        start=1
        end=32
        step=1
    fi
}

############################################################
# RUN COMMAND                                              #
############################################################

run_command()
{
    CMD=$1
    shift
    
    case "$CMD" in
        "check") 
            opt_list_short="hvb:g:" ; 
            opt_list="help,verbose,block-size:,grid-size:" ; 
            check_option $@
            check_args $ARGS
            run_check $@ ;;
        "measure" ) 
            opt_list_short="hvn:o:b:g:p::" ; 
            opt_list="help,rerun,verbose,nrep:,output:,block-size:,grid-size:,plot::" ; 
            check_option $@
            check_args $ARGS
            run_measure $@ ;;
        "export" )
            opt_list_short="ho:i:" ; 
            opt_list="help,output:,input:" ; 
            check_option $@
            check_args $ARGS
            run_export $@ ;;
       
        -h|--help) usage ;; 
        "") echo "OmniBench: need command"; usage ;;
        *) echo "OmniBench: $CMD is not an available command"; usage ;;
    esac
}



############################################################
# EXPORT COMMAND                                           #
############################################################

run_export()
{
    check_option $@
    if [[ -z "$input" ]]; then
        echo "Need an input"
        usage
    fi
    if [[ -z "$output" ]]; then
        output="${input%.csv}.md"
    fi
    echo "$(get_gpu_info)" > $output
    echo "" >> $output
    echo "## CSV Data" >> $output
    csv_to_mdtable $output $input
}

format_cell() 
{
    local cell="$1"
    local length="$2"
    printf "| %-*s " "$length" "$cell"
}

csv_to_mdtable()
{
    declare -a col_lengths

    while IFS=',' read -r -a columns; do
        for i in "${!columns[@]}"; do
            len=${#columns[i]}
            if [[ -z "${col_lengths[i]}" ]] || (( len > col_lengths[i] )); then
                col_lengths[i]=$len
            fi
        done
    done < "$2"

    while IFS=',' read -r -a columns; do
        if [ "$header" != "1" ]; then
            for i in "${!columns[@]}"; do
                format_cell "${columns[i]}" "${col_lengths[i]}" >> "$1"
            done
            printf "|\n" >> "$1"
            
            for length in "${col_lengths[@]}"; do
                printf "| %-${length}s " "---" >> "$1"
            done
            printf "|\n" >> "$1"
            
            header=1
        else
            for i in "${!columns[@]}"; do
                format_cell "${columns[i]}" "${col_lengths[i]}" >> "$1"
            done
            printf "|\n" >> "$1"
        fi
    done < "$2"
}

get_gpu_info()
{
    rocminfo_output=$(rocminfo)
    agent_info=$(echo "$rocminfo_output" | awk '/Agent 2/,/Done/')
    gpu_info=$(echo "$agent_info" | awk '
    /^Agent [0-9]+/ { agent = $3 }
    /^  Name: / { name = $2 }
    /^  Marketing Name: / { marketing_name = substr($0, index($0,$3)) }
    /^  Cache Info:/ { cache_info = 1; next }
    /^  Cacheline Size: / { cacheline_size = $3 }
    /^  Compute Unit: / { compute_unit = $3 }
    /^  SIMDs per CU: / { simds_per_cu = $4 }
    /^  Wavefront Size: / { wavefront_size = $3 }
    /^  Workgroup Max Size:/ { workgroup_max_size = $4 }
    /^  / && cache_info == 1 { if ($1 == "L1:" || $1 == "L2:" || $1 == "L3:") { cache_sizes = cache_sizes "- "$1" " $2" " "KB" "\n" } }
    /^$/ { cache_info = 0 }

    END {
    print "## GPU Information:"
    print ""
    print "GPU Name:             " name
    print ""
    print "Marketing Name:       " marketing_name
    print ""
    print "Compute Unit:         " compute_unit
    print ""
    print "SIMDs per CU:         " simds_per_cu
    print ""
    print "Wavefront Size:       " wavefront_size
    print ""
    print "Workgroup Max Size:   " workgroup_max_size
    print ""
    print "Cacheline Size:       " cacheline_size " bytes"
    print ""
    print "Cache Info:"  
    print cache_sizes  
    }')

    echo "$gpu_info"
}

############################################################
# MEASURE COMMAND                                          #
############################################################

run_measure()
{
    METRICS="DurationNs MeanOccupancyPerCU MeanOccupancyPerActiveCU GPUBusy Wavefronts L2CacheHit SALUInsts VALUInsts SFetchInsts"
    ROCPROF_OUTPUT=$TMPDIR/results.csv
    ROCPROF_INPUT=./config/input.txt

    create_output_csv_file "Kernel Optimisation ProblemSize BlockSize GridSize MyDurationMin MyDurationMed Scalability $METRICS"

    log_printf "=== Benchmark $BENCH for $KERNEL ($OPT) with size: $PB_SIZE"

    case "$BENCH" in
        "basic") 
              run_basic ;;
        "blockSizeVar") 
              run_blockSizeVar ;;
        "gridSizeVar") 
              run_gridSizeVar ;;
        *) echo "OmniBench: "$BENCH" is not an available benchmark"; usage ;;
    esac
}

run_blockSizeVar()
{
    build_driver measure $KERNEL $OPT
    for block_size in $(seq $start $step $end) ; do
        grid_size=$((($PB_SIZE + $block_size - 1) / $block_size))
        echo_run "measure" $KERNEL $OPT $PB_SIZE $block_size $grid_size "($(( (block_size * 100) / end ))%)"
        set_call_args $PB_SIZE $block_size $grid_size $nrep
        rocprof_app
        extract_rocprof_metrics_to $output
    done
    echo
    echo "Result saved in '$output'"

    if [ "$plot" == 1 ]; then
        log_printf "=== Plot generation . . ."
        python3 ./python/plot_from_csv.py "$output" "BlockSize" "DurationNs" --save_plot "$plot_file"
        echo "Plot created in file $plot_file"
    fi
}

run_gridSizeVar()
{
    build_driver measure $KERNEL $OPT
    start=40
    end=2
    step=600
    for grid_size in $(seq $start $step $end) ; do
        echo_run "measure" $KERNEL $OPT $PB_SIZE $block_size $grid_size "($(( (grid_size * 100) / end ))%)"
        set_call_args $PB_SIZE $block_size $grid_size $nrep
        rocprof_app
        extract_rocprof_metrics_to $output
    done
    echo
    echo "Result saved in '$output'"
}

run_basic()
{
    build_driver measure $KERNEL $OPT
    echo_run "measure" $KERNEL $OPT $PB_SIZE $block_size $grid_size
    set_call_args $PB_SIZE $block_size $grid_size $nrep
    rocprof_app
    extract_rocprof_metrics_to $output
    echo
    echo "Result saved in '$output'"
}

call_driver()
{
    echo $BIN_PATH/measure $PB_SIZE $BLOCK_DIM $GRID_DIM $NB_REP
}

rocprof_app()
{
    eval_verbose rocprof -o $ROCPROF_OUTPUT -i $ROCPROF_INPUT --timestamp on --stats --basenames on $(call_driver) 
}

create_output_csv_file()
{
    if [[ -z $output ]]; then
        output="$RESULTDIR/output_$(date +%F-%T).csv"
    fi

    if [[ ! -f "$output" ]]; then
        formatted_header=$(echo "$1" | tr ' ' ',')
        echo "$formatted_header" > "$output"
    fi
}

extract_rocprof_metrics_to()
{
    echo -n "$KERNEL" >> $1
    echo -n ",$OPT" >> $1
    echo -n ",$PB_SIZE" >> $1
    echo -n ",$BLOCK_DIM" >> $1
    echo -n ",$GRID_DIM" >> $1

    time_min=$(grep 'time_min:' "$TMPDIR/measure_tmp.out" | cut -d ' ' -f 2)
    time_med=$(grep 'time_med:' "$TMPDIR/measure_tmp.out" | cut -d ' ' -f 2)
    stability=$(grep 'stability:' "$TMPDIR/measure_tmp.out" | cut -d ' ' -f 2)
    echo -n ",$time_min" >> $1
    echo -n ",$time_med" >> $1
    echo -n ",$stability" >> $1

    for metric in $METRICS; do
        col_num=$(head -1 $ROCPROF_OUTPUT| tr ',' '\n' | nl -v 0 | grep $metric | awk '{print $1}')
        metric_value=$(awk -F, -v col=$((col_num+1)) 'NR > 0 {print $col}' $ROCPROF_OUTPUT | tail -n 1)
        echo -n ",$metric_value" >> $1
    done
    echo "" >> $1
}


############################################################
# CHECK COMMAND                                            #
############################################################

call_driver_check()
{
    echo $BIN_PATH/check $PB_SIZE $BLOCK_DIM $GRID_DIM $CHECK_OUT_FILE $ONLY_GPU
}

run_check()
{
    build_driver check "$KERNEL" "$OPT"
    set_call_args $PB_SIZE $block_size $grid_sizes
    eval $(call_driver_check)
}

############################################################
# UTILS                                                    #
############################################################

check_error()
{
  err=$?
  if [ $err -ne 0 ]; then
    echo -e "OmniBench: error in $0\n\t$1 ($err)"
    echo "Script exit."
    exit 1
  fi
}

eval_verbose()
{
  if [ "$verbose" == 1 ]; then
    eval $@
  elif [ "$verbose" == 0 ]; then
    eval $@ > /dev/null
  fi
}

current_datetime() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
}

log_printf() {
    local datetime=$(current_datetime)
    local message="$*"
    echo "[$datetime] $message" >> output.log
    echo "$message"
}

echo_run()
{
    echo -ne "\r=== Run $1 $2 $3 (size: $4, blockDim: $5, gridDim: $6) $7"
}

build_driver()
{
  log_printf "=== Compilation $1 $2 ($3) . . ."
  eval_verbose make $1 KERNEL=$2 OPT=$3 DIM=$DIM
  eval_verbose make kernel KERNEL=$2 OPT=$3 -B
  check_error "compilation failed"
}

set_call_args()
{
    PB_SIZE=$1
    BLOCK_DIM=$2
    GRID_DIM=$3
    NB_REP=$4
    CHECK_OUT_FILE=$4
    ONLY_GPU=$5
}

log_printf "================ START ================"

WORKDIR=`realpath $(dirname $0)`
TMPDIR="$WORKDIR/tmp"
RESULTDIR="$WORKDIR/results"
mkdir -p $TMPDIR
mkdir -p $RESULTDIR
cd $WORKDIR
run_command "$@"

rm tmp -rf
log_printf "================= END ================="
