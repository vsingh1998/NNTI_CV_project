# TASK III: Semi-supervised learning using Pseudo Labeling + Siamese Network

## Trained models

### CIFAR-10

- **For 4000 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-10/task3_c10_4k.pth
  ```

- **For 250 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-10/task3_c10_250.pth
  ```

### CIFAR-100

- **For 10000 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-100/task3_c100_10k.pth
  ```

- **For 2500 labeled samples**
  ```
  Directory: ./trained_models/CIFAR-100/task3_c100_2500.pth
  ```


## Usage

### For training

- CIFAR-10 (4000 labeled)
  ```
  python3 main.py --dataset cifar10 --num-labeled 4000 --total-iter 400*512 --iter-per-epoch 512 --threshold 0.75 --model-depth 16 --model-width 8 --dropout 0.3 --modelpath ../Task1/trained_models/CIFAR-10/4000/task1_c10_4k_t75.pth
  ```

- CIFAR-10 (250 labeled)
   ```
  python3 main.py --dataset cifar10 --num-labeled 250 --total-iter 200*256 --iter-per-epoch 256 --threshold 0.75 --model-depth 16 --model-width 8 --dropout 0.0 --modelpath ../Task1/trained_models/CIFAR-10/250/task1_c10_250_t75.pth
  ```
  
- CIFAR-100 (10000 labeled)
  ```
  python3 main.py --dataset cifar100 --num-labeled 10000 --total-iter 200*512 --iter-per-epoch 512 --threshold 0.75 --model-depth 16 --model-width 8 --dropout 0.3 --modelpath ../Task1/trained_models/CIFAR-100/10000/task1_c100_10k_t75.pth
  ```

- CIFAR-100 (2500 labeled)
   ```
  python3 main.py --dataset cifar100 --num-labeled 2500 --total-iter 200*256 --iter-per-epoch 256 --threshold 0.75 --model-depth 16 --model-width 8 --dropout 0.0 --modelpath ../Task1/trained_models/CIFAR-100/2500/task1_c100_2500_t75.pth
  ```

  ## Steps for testing

- Create a test dataset of type ```torch.utils.data.Dataset```
- Call one of the functions from ```test.py``` including ```test_cifar10``` or ```test_cifar100``` and pass the arguments ```testdataset``` and ```filepath```. For example:
  ```
    test_cifar10(testdataset, filepath='./trained_models/CIFAR-10/task3_c10_250.pth')
  ```
- The function will return the logits of the test dataset.