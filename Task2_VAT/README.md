# TASK II: Semi-supervised classification using Virtual Adversarial Training

## Trained models

### CIFAR-10

- **For 4000 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-10/4000/{model_name}
  ```
  - *task2_c10_4k.pth*: model for CIFAR-10 with 4000 labeled samples 

- **For 250 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-10/250/{model_name}
  ```
  - *task2_c10_250.pth*: model for CIFAR-10 with 250 labeled samples 

### CIFAR-100

- **For 10000 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-100/10000/{model_name}
  ```
  - *task2_c100_10k.pth*: model for CIFAR-100 with 10000 labeled samples 
  - 
- **For 2500 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-100/2500/{model_name}
  ```
  - *task2_c100_2500.pth*: model for CIFAR-100 with 2500 labeled samples


## Usage

### For training

- CIFAR-10 (4000 labeled)
  ```
  python3 main.py --dataset cifar10 --num-labeled 4000 --total-iter 150*800 --iter-per-epoch 800 --model-depth 16 --model-width 8 --vat-xi 1e-6 --vat-eps 6.0 --lr 0.001
  ```
- CIFAR-10 (250 labeled)
   ```
  python3 main.py --dataset cifar10 --num-labeled 250 --total-iter 100*128 --iter-per-epoch 128 --model-depth 16 --model-width 8 --vat-xi 1e-6 --vat-eps 6.0 --lr 0.001
  ```

- CIFAR-100 (10000 labeled)

  ```
  python3 main.py --dataset cifar100 --num-labeled 10000 --total-iter 150*800 --iter-per-epoch 800 --model-depth 16 --model-width 8 --vat-xi 1e-6 --vat-eps 6.0 --lr 0.001
  ```
- CIFAR-100 (2500 labeled)
   ```
  python3 main.py --dataset cifar10 --num-labeled 2500 --total-iter 100*256 --iter-per-epoch 256 --model-depth 16 --model-width 8 --vat-xi 1e-6 --vat-eps 6.0 --lr 0.001
  ```

  ## Steps for testing

- Create a test dataset of type *torch.utils.data.Dataset*
- Call one of the functions from *test.py* including *test_cifar10* or *test_cifar100* and pass the arguments *testdataset* and *filepath*. For example:
  ```
    test_cifar10(testdataset, filepath='./trained_models/CIFAR-10/250/task1_c10_250_t60.pth')
  ```
- The function will return the logits of the test dataset.