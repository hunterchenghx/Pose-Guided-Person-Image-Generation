
if [ ! -d ./data/DF_img_pose/DF_test_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_test_data.zip
    unzip DF_test_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_256x256PoseRCV_Mask_test_sparse_partBbox37_maskR4R8_roi10Complete DF_test_data
    rm -f DF_test_data.zip
    cd -
fi

#######################################################################
################################ Testing ##############################
gpu=1
D_arch='DCGAN'

model_dir=/media/cheng/marvel/Pose-Guided-Person-Image-Generation/DF
start_step=0
ckpt_path=${model_dir}'/model.ckpt-'${start_step}

## Make sure dataset name appear in  --dataset  (i.e. 'Market' or 'DF')
python main_hunter.py --dataset=DF_img_pose/DF_test_data \
             --img_H=256  --img_W=256 \
             --batch_size=1 \
             --is_train=False \
             --model=11 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=${start_step} --ckpt_path=${ckpt_path} \
             --test_one_by_one=True \
## Score
#stage_num=1
#python score.py ${stage_num} ${gpu} ${model_dir} 'test_result'
#python score_mask.py ${stage_num} ${gpu} ${model_dir} 'test_result'
