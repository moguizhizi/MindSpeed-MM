# 快速上手

## wan2.1预训练（T2V 1.3B）

环境安装请[参考](https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/wan2.1)

1. 权重转换：

    ```shell
    python examples/wan2.1/convert_ckpt.py --source_path <./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/> --target_path <./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/> --mode convert_to_mm
    ```

    * 启动脚本中修改LOAD_PATH为权重转换后的路径 (./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/)，SAVE_PATH为保存路径

2. 特征提取：

    * 修改`examples/wan2.1/feature_extract/data.txt`文件，其中每一行表示个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔。

    * 修改`examples/wan2.1/feature_extract/model_wan.json`修改为下载的权重所对应路径（包括vae, tokenizer, text_encoder）

    ```bash
    bash examples/wan2.1/feature_extract/feature_extraction.sh
    ```

3. 预训练：

    * feature_data.json中修改tokenizer权重路径

    ```bash
    bash examples/wan2.1/1.3b/t2v/pretrain.sh
    ```

## Qwen2.5-VL-3B预训练

环境安装请[参考](https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/qwen2.5vl)

1. 权重转换：
    将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2.5-VL-3B-Instruct`目录下

    ```bash
    mm-convert  Qwen2_5_VLConverter hf_to_mm \
    --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
    --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
    --cfg.parallel_config.llm_pp_layers [[36]] \
    --cfg.parallel_config.vit_pp_layers [[32]] \
    --cfg.parallel_config.tp_size 1
    # 其中：
    # mm_dir: 转换后保存目录
    # hf_dir: huggingface权重目录
    # 其余无需修改
    ```

    * 修改`examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-3B-Instruct`进行区分

    ```shell
    LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
    ```

2. 数据集准备：

    (1)下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中

    (2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下;

    (3)运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py;

    ```
    $playground
    ├── data
        ├── COCO2017
            ├── train2017

        ├── llava_instruct_150k.json
        ├── mllm_format_llava_instruct_data.json
        ...
    ```

    注意`data.json`中`dataset_param->basic_parameters->max_samples`的配置，会限制数据只读`max_samples`条，这样可以快速验证功能。如果正式训练时，可以把该参数去掉则读取全部的数据。

    在数据构造时，对于包含图片的数据，需要保留`image`这个键值。

    ```python
    {
    "id": your_id,
    "image": your_image_path,
    "conversations": [
        {"from": "human", "value": your_query},
        {"from": "gpt", "value": your_response},
    ],
    }
    ```

3. 微调

    * 根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path` (权重转换前的权重路径"./ckpt/hf_path/Qwen2.5-VL-3B-Instruct")、`dataset_dir` ("./data")、`dataset` ("./data/mllm_format_llava_instruct_data.json")等字段。

    ```shell
    bash examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh
    ```
