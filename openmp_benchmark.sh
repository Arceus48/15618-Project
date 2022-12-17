#!/bin/bash


export OMP_NUM_THREADS=1

echo "1 threads"

for i in 1 2 3
do
  for j in 750 1500 3000 4500 6000
  do
    echo $j
    ./src/gradient_domain ./img/${j}_ambient.jpg ./img/${j}_flash.jpg 10 0.5 2 2 0.0005 500
  done
done

export OMP_NUM_THREADS=2

echo "2 threads"

for i in 1 2 3
do
  for j in 750 1500 3000 4500 6000
  do
    echo $j
    ./src/gradient_domain ./img/${j}_ambient.jpg ./img/${j}_flash.jpg 10 0.5 2 2 0.0005 500
  done
done

export OMP_NUM_THREADS=4

echo "4 threads"

for i in 1 2 3
do
  for j in 750 1500 3000 4500 6000
  do
    echo $j
    ./src/gradient_domain ./img/${j}_ambient.jpg ./img/${j}_flash.jpg 10 0.5 2 2 0.0005 500
  done
done

export OMP_NUM_THREADS=8

echo "8 threads"

for i in 1 2 3
do
  for j in 750 1500 3000 4500 6000
  do
    echo $j
    ./src/gradient_domain ./img/${j}_ambient.jpg ./img/${j}_flash.jpg 10 0.5 2 2 0.0005 500
  done
done

export OMP_NUM_THREADS=16

echo "16 threads"

for i in 1 2 3
do
  for j in 750 1500 3000 4500 6000
  do
    echo $j
    ./src/gradient_domain ./img/${j}_ambient.jpg ./img/${j}_flash.jpg 10 0.5 2 2 0.0005 500
  done
done