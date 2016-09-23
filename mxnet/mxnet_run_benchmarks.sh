#!/bin/bash

CMBS=16
CNB=100

LMBS=128
LNB=100

FMBS=64
FNB=100

for i in "$@"
do
case $i in
	-cm=*|--cnn_mini_batch_size=*)
	CMBS="${i#*=}"
	shift	# past argument=value
	;;
	-cn=*|--cnn_num_batch=*)
	CNB="${i#*=}"
	shift	# past argument=value
	;;

	-lm=*|--lstm_mini_batch_size=*)
	LMBS="${i#*=}"
	shift	# past argument=value
	;;
	-ln=*|--lstm_num_batch=*)
	LNB="${i#*=}"
	shift	# past argument=value
	;;
	
	-fm=*|--fcn_mini_batch_size=*)
	FMBS="${i#*=}"
	shift	# past argument=value
	;;
	-fn=*|--fcn_num_batch=*)
	FNB="${i#*=}"
	shift	# past argument=value
	;;

	--default)
	DEFAULT=YES
	shift	# past argument=value
	;;
	*)
		# unknown option
	;;
esac
done

sudo rm -f output_alexnet.log
sudo rm -f output_resnet.log
sudo rm -f output_fcn5.log
sudo rm -f output_fcn8.log
sudo rm -f output_lstm32.log
sudo rm -f output_lstm64.log

python cnn/cnn.py --batch-size ${CMBS} --num-batch ${CNB} --network alexnet 2>&1 | tee output_alexnet.log
python cnn/cnn.py --batch-size ${CMBS} --num-batch ${CNB} --network resnet 2>&1 | tee output_resnet.log
python fcn/fcn5.py --batch-size ${FMBS} --num-batch ${FNB} 2>&1 | tee output_fcn5.log
python fcn/fcn8.py --batch-size ${FMBS} --num-batch ${FNB} 2>&1 | tee output_fcn8.log
python rnn/lstm_bucketing.py --batch-size ${LMBS} --num-batch ${LNB} --seq-len 32 --data-path "..\cntk\rnn\PennTreebank\Data" 2>&1 | tee output_lstm32.log
python rnn/lstm_bucketing.py --batch-size ${LMBS} --num-batch ${LNB} --seq-len 64 --data-path "..\cntk\rnn\PennTreebank\Data" 2>&1 | tee output_lstm64.log	
