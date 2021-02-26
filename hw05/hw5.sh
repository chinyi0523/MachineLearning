echo $1
echo $2
mkdir $2
python3 confusion.py -test $1 model_train_VGG1.pkl $2
python3 hw5_explainable.py -Saliency $1 $2
python3 hw5_explainable.py -Filter $1 $2
python3 hw5_explainable.py -Lime $1 $2
python3 hw5_explainable.py -Dream $1 $2
