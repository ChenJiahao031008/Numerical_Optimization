#! /usr/bin/env bash

function build_project(){
    mkdir -p build && cd build
    cmake ..
    make -j4
    cd ../
}

function run_project(){
    cd build/
    ./AMao
    cd ../
}

function test_project(){
    cd build/unit_testing/
    ./opt_sgd_test
    cd ../
}

function clean_project(){
    rm -rf log/
    rm -rf build/*
}


function main() {
    local cmd="$1"
    shift
    case "${cmd}" in
        build)
            build_project
            ;;
        test)
            test_project
            ;;
        run)
            run_project
            ;;
        build_and_test)
            build_project && test_project
            ;;
        build_and_run)
            build_project && run_project
            ;;
        clean)
            clean_project
            ;;

    esac
}

main "$@"
