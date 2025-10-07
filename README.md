## SecureML_OPTEE
Experimentation of ML Model Execution within OPTEE

Currently TA Stack and TA Heap are set to 1 MB and 4 MB respectively.

# Configure TA Stack
To configure the TA Stack with custom memory size, run this command 
```
cd <respective_repo>/ta/include/
#Here the command that sets the TA_STACK_SIZE to 1 MB. For other settings, change the value of 1 * 1024 * 1024
sed -i 's/\(#define TA_STACK_SIZE\s\+\)([^)]*)/\1(1 * 1024 * 1024)/' user_ta_header_defines.h 
```
# Configure TA Data/Heap
To configure the TA Data/Heap with custom memory size, run this command 
```
cd <respective_repo>/ta/include/
# Here the command that sets the TA_DATA_SIZE to 4 MB. For other settings, change the value of 4 * 1024 * 1024
sed -i 's/\(#define TA_DATA_SIZE\s\+\)([^)]*)/\1(4 * 1024 * 1024)/' user_ta_header_defines.h 
```
# Inference
To run the inference using Lenet using Darknetz
```
 darknetp classifier predict -pp_start 4 -pp_end 10 datasets/darknetz/cfg/mnist.dataset  datasets/darknetz/cfg/mnist_lenet.cfg datasets/darknetz/models/mnist/mnist_lenet.weights  datasets/darknetz/data/mnist/images/t_00002_c4.png
```



To run the inference using MobileNet-V1 using SmartZone (TinyLib)
```
tinylib classifier predict datasets/SmartZone/cfg/imagenet1k.data  datasets/SmartZone/cfg/mobilenet_v1.cfg  datasets/SmartZone/models/mobilenet_v1.weights  datasets/SmartZone/data/cat.jpg  -mov_size 262144
```
