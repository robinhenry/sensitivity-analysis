#!/usr/bin/env bash

for sc in 0.5 1. 0. 0.2
do
  for x_i in 10 11 5
  do

    python main.py $x_i $sc

  done
done