GPUID=0
OUTDIR=outputs/split_MNIST_incremental_domain
REPEAT=10
MODE=$1
mkdir -p outputs/split_MNIST_incremental_domain
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.001 --offline_training  | tee ${OUTDIR}/Offline.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.001   --mode $1                  | tee ${OUTDIR}/Adam_${MODE}.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.01        --mode $1                | tee ${OUTDIR}/Adagrad_${MODE}.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name EWC_online_mnist --lr 0.001 --reg_coef 700 --mode $1     | tee ${OUTDIR}/EWC_online_${MODE}.log
#python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name EWC_mnist        --lr 0.001 --reg_coef 100 --mode $1     | tee ${OUTDIR}/EWC_${MODE}.log
python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name L2  --lr 0.001 --reg_coef 0.5   --mode $1   | tee ${OUTDIR}/L2_${MODE}.log