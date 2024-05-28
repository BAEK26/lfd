epochs=100
file_name=test
echo "run model in $epochs epochs and generate scenario at scenarios/$file_name.csv"
echo "start running..."
python model.py --epochs $epochs --file_name $file_name