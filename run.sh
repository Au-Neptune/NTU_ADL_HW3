mkdir "tmp"
echo "[step log] make tmp folder"
python ./inference.py --model_path $1 --peft_path $2 --data $3 --output_path ./tmp/prediction.json
echo "[step log] predict done"
mv ./tmp/prediction.json $4
echo "[step log] ALL DONE"