#!/bin/bash

scores=$1
file=$2
outputFile=$3
gens=$4

cmd="{ command time -f "%U,%S" timeout 16h ./search $scores $file $gens 0 out.txt; } 2>> $outputFile "
echo "$cmd"
eval "$cmd"