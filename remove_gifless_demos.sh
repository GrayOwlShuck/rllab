#!/bin/bash


#for i in "${test_ids[@]}"

for i in {1..1000}
do 
    if [   -f data/oneobj_push_demos_filter_vision/train/"$i".pkl ]; then
        if [  ! -f data/oneobj_push_demos_filter_vision/train/crop_object_"$i"/cond7.samp0.gif ]; then
            echo "$i"
            #rm -r data/fixed_push_demos_filter_vision/train/object_"$i"
            #rm data/fixed_push_demos_filter_vision/train/"$i".pkl
        fi
    fi
done
