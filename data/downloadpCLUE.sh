# download pCLUE dataset train set
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_1.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_2.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_3.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_4.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_5.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_6.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_7.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_8.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_train_9.json

cat pCLUE_train_1.json pCLUE_train_2.json pCLUE_train_3.json pCLUE_train_4.json pCLUE_train_5.json pCLUE_train_6.json pCLUE_train_7.json pCLUE_train_8.json pCLUE_train_9.json > pCLUE_train.json

wc -l pCLUE_train.json

# download pCLUE dataset dev set
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_dev.json
wc -l pCLUE_dev.json

# download pCLUE dataset test set
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_test_public_1.json
wget  https://raw.githubusercontent.com/CLUEbenchmark/pCLUE/main/datasets/pCLUE_test_public_2.json

cat pCLUE_test_public_1.json pCLUE_test_public_2.json > pCLUE_test_public.json

wc -l pCLUE_test_public.json