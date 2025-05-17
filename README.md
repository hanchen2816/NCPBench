# NCPBench
# 数字约束问题
## 一：构建的数据集
### 1.1、构建的中文（1000条），英文（1056条）具体数字约束问题数据集。
```
中文：data/chinese/简单生成任务.txt
英文：data/english/generate_Q.txt
```
### 1.2、构建的中文（1000条），英文（1056条）范围数字约束问题数据集。
```
中文：data/range_Q/CH.txt
英文：data/range_Q/EN.txt
```
### 1.3、构建的大模型微调数据集
```
中文：data/fine_turning/constrain_numeric_train.json
```
## 二：测试不同大模型的数字约束生成能力
### 2.1、强约束问题
#### 2.1.1、测试大模型对于不同类型数字的约束问题的能力
```
# 运行中文测试
python run_test.py --language cn --data_dir data/chinese --model_name gpt-4o-mini-2024-07-18 --api_key your_api_key --output_dir test_results_20240318

# 运行英文测试
python run_test.py --language en --data_dir data/english --model_name gpt-4o-mini-2024-07-18 --api_key your_api_key --output_dir test_results_20240318
```
#### 2.1.2、测试大模型对于1-100每个数字的约束问题的能力
```
# 运行完整范围（1-100）的中文测试
python run_test1-100.py --language cn --data_dir data/chinese --model_name gpt-4o-mini-2024-07-18 --api_key your_api_key --output_dir test_results_20240318

# 运行部分范围（如1-50）的英文测试
python run_test1-100.py --language en --data_dir data/english --model_name gpt-4o-mini-2024-07-18 --api_key your_api_key --output_dir test_results_20240318 --start_num 1 --end_num 50

```

### 2.2、弱(范围)约束问题
#### 2.2.1、
```
# 运行中文范围约束测试
python run_range_test.py --language cn --data_dir data/chinese --model_name qwen1.5-72b-chat --api_key your_api_key --output_dir test_results_20240318

# 运行英文范围约束测试
python run_range_test.py --language en --data_dir data/english --model_name qwen1.5-72b-chat --api_key your_api_key --output_dir test_results_20240318
```

## 三：提升大模型面对数字约束问题时候的能力
`improve/`文件夹下
### 3.1、提示工程
#### 在每次输入prompt给大模型的时候，在后面加上需要生成的约束问题的示例
```
prompt_project.py
```
### 3.2、step-by-step
#### 利用提示，将每次让大模型生成一个词，并且提醒大模型已经生成的字数以及还需要生成的剩余字数
```
improve/step_by_step.py
```
### 3.3、微调
#### 接着我们设想，这种step-by-step的方式能否提高大模型对于制定生成数字个数约束问题的理解能力，于是，我们将3.2中的测试结果构建了962条微调数据集
```
improve/fine_turning.py
```
### 3.4、moe思维链
#### 利用思维链
