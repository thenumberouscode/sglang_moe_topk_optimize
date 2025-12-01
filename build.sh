#!/bin/bash

set -e  # 遇到错误立即退出

case "$1" in
    "build")
        set -x
        rm -rf build
        rm -rf softmax_extension.cpython*
        python setup.py build_ext --inplace
        mkdir -p perfs
        ;;
    "perf")
        if [ -z "$2" ]; then
            echo "Usage: $0 perf <output_name>"
            exit 1
        fi
        ncu --set full --target-processes all -f -o "perfs/$2" python sglang_bench_softmax.py
        ;;
    "diff")
        export IS_DIFF=1
        if [ ! -z "$2" ]; then
            echo "log level: $2"
            export LOG_LEVEL="$2"
        fi
        python sglang_bench_softmax.py
        ;;
    *)
        export IS_TRI=1
        if [ ! -z "$2" ]; then
            echo "log level: $2"
            export LOG_LEVEL="$2"
        fi
        python sglang_bench_softmax.py
        ;;
esac
 
