echo $1
echo $2
mkdir checkpoints
mkdir checkpoints_save
python3 hw9.py -train $1 $2 -improved