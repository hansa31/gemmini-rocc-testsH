#!/usr/bin/env bash

if [ ! -d "build" ] ; then
    autoconf && \
        mkdir build && cd build && \
        ../configure &&
        cd ..

    if [ $? -ne 0 ] ; then
        echo $0 failed
        exit 1
    fi
fi

cd build

if [[ $(which riscv64-unknown-linux-gnu-gcc) ]] ; then
    # Force remake of all targets so previous build files are replaced
    make -B -j $@
else
    # Force remake for baremetal build as well
    make -B -j BAREMETAL_ONLY=1 $@
fi

