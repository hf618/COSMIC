#! /bin/bash
data_root='/home/hfd24/TPTs/dataset/tta_data'
#testsets=I/A/V/R/K
testsets=flower102/dtd/pets/cars/ucf101/caltech101/food101/sun397/aircraft/eurosat
# flower102/dtd/pets/cars/ucf101/caltech101/food101/sun397/aircraft/eurosat
ctx_init=a_photo_of_a

arch=ViT-B/16 # RN50 | ViT-B/16 | align-base
dino_size=l # l | b | s
center_type_clip=attn # default | ema | attn
center_type_dino=default

current_time=$(date +"%Y_%m_%d_%H_%M_%S")

for value in 1 2 4 8 16 32
do
  python ./cosmic_main.py ${data_root} --test_sets ${testsets}   \
    -a ${arch} --ctx_init ${ctx_init} --text_prompt tip_cupl  \
    --gpu 0 --beta 5.5 --config configs_l  --seed 0 \
    --selection_p 0.1 -b 8 --DINO_size ${dino_size} --center_type_clip ${center_type_clip} --center_type_dino ${center_type_dino} \
    --r 0.2 --DINO_Cache_shot 8 --CLIP_Cache_shot ${value} --use_clip_cache --DINOv2 --use_clip_clique --use_dino_clique \
    --mac_step 100 --target_avg_degree 5.0 --inrease_t \
    -p 1000 --log_time ${current_time} \
    --use_log --clip_is_DMN --is_SOG
done
