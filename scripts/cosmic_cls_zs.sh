######################### zero shot generalization to distribution shift. 
data_root='/home/hfd24/TPTs/dataset/tta_data'
#testsets=I/A/V/R/K
testsets=K
# flower102/dtd/pets/cars/ucf101/caltech101/food101/sun397/aircraft/eurosat

#arch=RN50  ViT-B/16 align-base
arch=ViT-B/16
dino_size=b
ctx_init=a_photo_of_a

for nshot in 0
do
  python ./cosmic_main.py ${data_root} --test_sets ${testsets}   \
  -a ${arch} --ctx_init ${ctx_init}  --memory_size 50 --text_prompt tip_cupl  --log camera_ready_dmn_tf_shift \
  --gpu 0 --n_shot ${nshot} --n_augview 0   --beta 5.5  --use_searched_param \
  --config configs_${dino_size} -p 1000 \
  --selection_p 0.1 -b 16 --DINO_size ${dino_size} --center_type 'default' \
  --DINOv2 --DINOv2_mem --use_MAC --use_MAC_logits --use_MAC_logits2 --Logits_wei_ablation --r 0.2 --seed 0 --inrease_t --DINO_shot 6 \
  --mac_step 10000 --target_avg_degree 4.0 --dino_standard 1 --is_DMN True --use_log
done
