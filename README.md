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
