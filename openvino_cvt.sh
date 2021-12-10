#!/bin/bash

if [ $# -ne 5 ]
then
  echo "Usage: ./openvino_cvt.sh <TF_SAVED_MODEL_DIR> <PRECISION> <OUTPUT_DIR> <OUTPUT_MODEL_NAME> <DEV>"
  exit 1
fi

python3 -m tf2onnx.convert --saved-model $1 --output temp.onnx && \
mo.py --input_model temp.onnx --output_dir $3 --model_name $4 --data_type $2 && \
rm temp.onnx && \
python3 verify_openvino.py $1 $3/$4 $5
