#!/bin/bash

for file in unprocessed/*; do
    output=${file/unprocessed/highfreq}
    output=${output/.mp3/.wav}
    ffmpeg -i "$file" -ac 1 -acodec pcm_u8 -sample_fmt u8 -ar 44100 "$output"
done
