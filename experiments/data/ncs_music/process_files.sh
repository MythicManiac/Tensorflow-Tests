#!/bin/bash

for file in unprocessed/*; do
    output=${file/unprocessed/processed}
    output=${output/.mp3/.wav}
    ffmpeg -i "$file" -ac 1 -acodec pcm_u8 -sample_fmt u8 -ar 4000 "$output"
done
