# WatME: Towards Lossless Watermarking Through Lexical Redundancy
Welcome to the repository for our ACL 2024 paper, "WatME: Towards Lossless Watermarking Through Lexical Redundancy" In this work, we introduce **WatME** (Watermarking with Mutual Exclusion), a novel approach leveraging linguistic prior knowledge of inherent lexical redundancy in LLMs' vocabularies to seamlessly integrate watermarks.

## WatME Method
This illustration highlights the advantages of the WatME (Watermarking with Mutual Exclusion) Method for lossless watermarking. The left panel displays a Vanilla large language model (LLM) that utilizes all available tokens during generation. The middle panel reveals a flaw in traditional KGW-watermarking approaches, which may indiscriminately assign all suitable tokens (e.g., 'ocean' and 'sea') to a red list, thus diminishing the expressiveness of the LLM. The right panel demonstrates how WatME addresses this issue by harnessing lexical redundancy. It applies a mutual exclusion rule to redundant tokens, ensuring that at least one appropriate token remains available (on the green list) during the watermarking process, thereby preserving the expressive power of LLMs.

<img width="348" alt="image" src="https://github.com/ChanLiang/WatME/assets/44222294/3f9569f0-a83b-4ad7-8c69-4bfa73c68a9b">

## Getting Started

#### 1. Set Up the Environment

Begin by setting up your Conda environment with the provided `env.txt` file, which will install all necessary packages and dependencies.

```bash
cd watermark
pip install -r env.txt
```
If you run into any missing packages or dependencies, please install them as needed.


#### 2. Explore the Redundancy in Lexical Space

Code for building and processing synonym clusters is located in the `synonym` directory, with the processed clusters saved in the `output` directory.

- **Buinding Clusters:**

  (a) Use LLMs, e.g. llama to infer synonyms.
   
  ```bash
  python3 build_cluster_llama.py
  ```

  (b) Use an external dictionary (e.g. youdao) to obtain synonyms.

   ```bash
   python3 get_synonmy_by_youdao.py
   python3 build_clusters.py
   ```

- **Post-processing:**
   ```bash
   python3 filter_cluster.py
   ```

#### 3. Exploit the Lexical Redundancy During Watermarking

The process is divided into two main parts: running experiments and parsing results. Ensure that the experimental data is downloaded into the `data` folder and the processed synonym clusters are placed in the `output` folder before proceeding.

- Run experiments with WatME.
Execute the following script to run experiments, replacing ${task} with the appropriate task name (e.g., truthfulqa).
```bash
bash scripts/${task}_llama2_run.sh 
```

- Parse result.
After running the experiments, parse the results by executing the script corresponding to the task.
```bash
bash scripts/parse_${task}.sh 
```

## Interactive Demo Visualization
You may try our demo, which is implemented using [Gradio](https://gradio.app). For details, please refer to the `demo` folder.

## Citing Our Work
If you find our work helpful in your research, please cite our paper:
```
@misc{chen2024watme,
      title={WatME: Towards Lossless Watermarking Through Lexical Redundancy}, 
      author={Liang Chen and Yatao Bian and Yang Deng and Deng Cai and Shuaiyi Li and Peilin Zhao and Kam-fai Wong},
      year={2024},
      eprint={2311.09832},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
