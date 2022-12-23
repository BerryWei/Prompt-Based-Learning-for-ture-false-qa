
python trainQA.py \
    --train_data ./data/true-false-qa-TRAIN.json \
    --test_data  ./data/true-false-qa-TEST.json \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --num_epoch 200 \
    --batch_size 16 \
    --lr 1.0e-4 \
    --with_plotting
