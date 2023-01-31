#!/bin/bash
folder=$1
for i in ${folder}/*.wav
do
      	ffmpeg -y -i "$i" -ar 32000 "./tmp_data/${i%.*}.wav"
done
