# TASK I: Semi-supervised learning using Pseudo Labeling

## Trained models

### CIFAR-10

- **For 4000 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-10/4000/{model_name}
  ```
  - *task1_c10_4k_t60.pth*: model for CIFAR-10 with 4000 labeled samples having a threshold of 0.60
  - *task1_c10_4k_t75.pth*: model for CIFAR-10 with 4000 labeled samples having a threshold of 0.75
  - *task1_c10_4k_t95.pth*: model for CIFAR-10 with 4000 labeled samples having a threshold of 0.95

- **For 250 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-10/250/{model_name}
  ```
  - *task1_c10_250_t60.pth*: model for CIFAR-10 with 250 labeled samples having a threshold of 0.60
  - *task1_c10_250_t75.pth*: model for CIFAR-10 with 250 labeled samples having a threshold of 0.75
  - *task1_c10_250_t95.pth*: model for CIFAR-10 with 250 labeled samples having a threshold of 0.95

### CIFAR-100

- **For 10000 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-100/10000/{model_name}
  ```
  - *task1_c100_10k_t60.pth*: model for CIFAR-100 with 10000 labeled samples having a threshold of 0.60
  - *task1_c100_10k_t75.pth*: model for CIFAR-100 with 10000 labeled samples having a threshold of 0.75
  - *task1_c100_10k_t95.pth*: model for CIFAR-100 with 10000 labeled samples having a threshold of 0.95

- **For 2500 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-100/2500/{model_name}
  ```
  - *task1_c100_2500_t60.pth*: model for CIFAR-100 with 2500 labeled samples having a threshold of 0.60
  - *task1_c100_2500_t75.pth*: model for CIFAR-100 with 2500 labeled samples having a threshold of 0.75
  - *task1_c100_2500_t95.pth*: model for CIFAR-100 with 2500 labeled samples having a threshold of 0.95


## Usage

### For training

- CIFAR-10 (4000 labeled)
  ```
  python3 main.py --dataset cifar10 --num-labeled 4000 --total-iter 250*800 --iter-per-epoch 800 --threshold 0.60/0.75/0.95 --model-depth 16 --model-width 8 --dropout 0.3
  ```
- CIFAR-10 (250 labeled)
   ```
  python3 main.py --dataset cifar10 --num-labeled 250 --total-iter 128*150 --iter-per-epoch 128 --threshold 0.60/0.75/0.95 --model-depth 16 --model-width 8 --dropout 0.0
  ```

  - CIFAR-100 (10000 labeled)

  ```
  python3 main.py --dataset cifar100 --num-labeled 10000 --total-iter 250*800 --iter-per-epoch 800 --threshold 0.60/0.75/0.95 --model-depth 16 --model-width 8 --dropout 0.3
  ```
- CIFAR-100 (2500 labeled)
   ```
  python3 main.py --dataset cifar100 --num-labeled 2500 --total-iter 256*150 --iter-per-epoch 256 --threshold 0.60/0.75/0.95 --model-depth 16 --model-width 8 --dropout 0.0
  ```

  ## Steps for testing

- Create a test dataset of type ```torch.utils.data.Dataset```
- Call one of the functions from ```test.py``` including ```test_cifar10``` or ```test_cifar100``` and pass the arguments ```testdataset``` and ```filepath```. For example:
  ```
    test_cifar10(testdataset, filepath='./trained_models/CIFAR-10/250/task1_c10_250_t60.pth')
  ```
- The function will return the logits of the test dataset.