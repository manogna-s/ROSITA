#!/bin/bash
MODEL=$1
TTA_METHOD=$2
GPU_ID=$3


CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --weak_OOD cifar10OOD --strong_OOD MNIST    --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --weak_OOD cifar10OOD --strong_OOD SVHN     --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --weak_OOD cifar10OOD --strong_OOD cifar100 --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --weak_OOD cifar10OOD --strong_OOD Tiny     --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --weak_OOD cifar100OOD --strong_OOD MNIST   --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --weak_OOD cifar100OOD --strong_OOD SVHN    --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --weak_OOD cifar100OOD --strong_OOD cifar10 --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --weak_OOD cifar100OOD --strong_OOD Tiny    --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 50000 --weak_OOD ImagenetCOOD --strong_OOD MNIST  --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 50000 --weak_OOD ImagenetCOOD --strong_OOD SVHN   --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 30000 --weak_OOD ImagenetROOD --strong_OOD MNIST  --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 30000 --weak_OOD ImagenetROOD --strong_OOD SVHN   --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 50000 --weak_OOD VisdaOOD --strong_OOD MNIST      --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 50000 --weak_OOD VisdaOOD --strong_OOD SVHN       --model ${MODEL} --tta_method ${TTA_METHOD}