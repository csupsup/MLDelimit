#!/bin/bash

for i in {1..5}
do
    slim -d RUNID=$i s3_slim_reps.slim
    echo "Run $i done"
done

echo "All runs completed"
