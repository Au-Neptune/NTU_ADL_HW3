# ADL HW3 - Taiwan LLaMa

## Project Description
This project is part of the NTU Applied Deep Learning course (Fall 2023) and focuses on Instruction Tuning (Classical Chinese). The objective is to fine-tuning Taiwan LLaMa model that can translate between Classical Chinese and Normal Chinese.

![Task Description](./images/Task%20Description.png)

Due to this project involves fine-tuning a large language model, I used QLora. QLora is a method that allows for training the weights of an LLM without the need to retrain the entire language model, essentially functioning as a "plugin" approach.

![QLora](./images/QLora.png)

for more information please refer to [ADL2023-HW3](./ADL2023-HW3.pdf)

## Windows環境建置
由於bitsandbytes無法在Windows上執行，故選用WSL虛擬系統進行。
### 安裝WSL
參考以下兩個教學安裝WSL，可選是否升級成WSL2

[【WSL】Windows Subsystem for Linux 安裝及基本配置！](https://learn.microsoft.com/zh-tw/archive/blogs/microsoft_student_partners_in_taiwan/wsltune)

[舊版 WSL 的手動安裝步驟](https://learn.microsoft.com/zh-tw/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)

安裝完畢後，可在開始頁面點擊Ubuntu開啟終端機
### 安裝Miniconda
參考 [MiniConda官網](https://docs.conda.io/projects/miniconda/en/latest/index.html) 進行安裝
按照Linux欄位安裝完成後，請注意需在~/.bashrc文件中輸入
```bash
export PATH="$PATH:/home/<your name>/miniconda3"
```
儲存後記得執行
```bash
source ~/.bashrc
```
完成後應該會在用戶名前面看到(BASE)字樣

### 啟動虛擬環境
```bash
conda create -n ADL_HW3 python=3.10
conda env list #查看環境列表
conda activate ADL_HW3 #進入環境
conda deactivate #退出環境
```
---
## Enviroments
```bash
pip install -r requirements.txt
```
## Quick Start
```bash
bash ./download.sh
```
```bash
bash ./run.sh <path to Taiwan-LLaMa-folder> ./adapter_checkpoint <path to input.json> <path to output.json>
```
---
## Training
### Start Training
for example:
```bash
python ./finetune.py \
--model_path ./Taiwan-LLM-7B-v2.0-chat \
--train_data ./data/train.json \
--eval_data ./data/public_test.json \
--epoch 1 \
--seq_length 1024 \
--log_freq 0.1 \
--output_dir ./paged
```
* `model_path`: Path to Taiwan-LLM-7B-v2.0-chat.
* `train_data`: Path to `train.json`.
* `eval_data`: Path to `public_test.json`. Will use to log ppl score.
* `output_dir`: The output directory where the config will be stored.

---
## Testing
```bash
python ./inference.py \
--model_path ./Taiwan-LLM-7B-v2.0-chat \
--peft_path ./paged \
--data ./data/private_test.json \
--output_path ./paged/prediction.json
```
* `model_path`: Path to Taiwan-LLM-7B-v2.0-chat.
* `peft_path`: Path to trained peft config.
* `data`: Path to `private_test.json`.
* `output_path`: The output path where the result will be stored（json format）.


## Final Report
The final report of this project provides a comprehensive overview of the project, including the following sections:

1. **LLM Tuning**

2. **LLM Inference Strategies**

For more detailed information, please refer to the [full report](./report.pdf).