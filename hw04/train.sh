echo "start training for 7 models"
python3 main2.py -train 35 ckpt_pre_35_cbow_0.model
echo "save ckpt_pre_cbow_35_0.model"
python3 main2.py -train 35 ckpt_pre_35_cbow_1.model
echo "save ckpt_pre_cbow_35_1.model"
python3 main2.py -train 30 ckpt_pre_30_cbow_0.model
echo "save ckpt_pre_cbow_30_0.model"
python3 main2.py -train 30 ckpt_pre_30_cbow_1.model
echo "save ckpt_pre_cbow_30_1.model"
python3 main2.py -train 25 ckpt_pre_25_cbow_0.model
echo "save ckpt_pre_cbow_25_0.model"
python3 main2.py -train 25 ckpt_pre_25_cbow_1.model
echo "save ckpt_pre_cbow_25_1.model"
python3 main2.py -train 20 ckpt_pre_20_cbow_0.model
echo "save ckpt_pre_cbow_20_0.model"
