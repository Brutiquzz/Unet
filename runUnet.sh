#!/bin/bash
  # normal cpu stuff: allocate cpus, memory
  SBATCH --ntasks=1 --cpus-per-task=10 --mem=24000M
  # we run on the gpu partition and we allocate 2 titanx gpus
  SBATCH -p gpu --gres=gpu:titanx:2
  #We expect that our program should not run langer than 4 hours
  #Note that a program will be killed once it exceeds this time!
  SBATCH --time=6:00:00

  #your script, in this case: write the hostname and the ids of the chosen gpus.
  a00552
  echo $CUDA_VISIBLE_DEVICES
  python Unet/Main.py
