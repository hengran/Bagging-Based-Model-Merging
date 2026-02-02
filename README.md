The code for the paper "Robust Training for General Text Embeddings via Bagging-Based Model Merging".

We extend our code to the following open‑source repositories for their valuable contributions:

- **[FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)** for providing general embedding models and training datasets.
- **[QwenLM/Qwen3‑Embedding](https://github.com/QwenLM/Qwen3-Embedding/tree/main)** for sharing the MTEB evaluation code.

# Download Training Data

## Eng-Text-Data
Download the [full-data](https://huggingface.co/datasets/cfli/bge-full-data) provided by [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).

**Note on Data Exclusion:** The original training data contains three datasets—Quora, SCIDOCS-RR, and ArguAna—for which MTEB provides only test or development splits without corresponding training data. To prevent potential data contamination, we exclude these three datasets from the Eng-Text-Data.

## General-Full-Data
Follow these steps to assemble the General-Full-Data:

1. **Download Eng-Text-Data** as described above.

2. **Download Multilingual Retrieval Datasets** from [bge-e5data](https://huggingface.co/datasets/cfli/bge-e5data) provided by [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), which includes:
   - DuReader
   - MIRACL
   - Mr. TyDi
   - T2-Ranking

3. **Download Code Retrieval Datasets** from [Cornstack](https://huggingface.co/collections/nomic-ai/cornstack) provided by [Nomic-ai](https://github.com/nomic-ai).
   - We use six programming languages: go-v1, java-v1, python-v1, javascript-v1, php-v1, and ruby-v1.
   - Due to the large scale of the original data, we sample approximately 100,000 examples from each language for training.

4. **Merge** all downloaded datasets to create the General-Full-Data.

# How to Train the General Text Embedding using Batch-Level Shuffling?
Take the Qwen3-0.6 for example:
```
cd Training
sh base_same_0.6B.sh
```
# How to Train the General Text Embedding using BOOM?
To balance training costs while promoting generalization, train *m* models using Batch-Level Shuffling on differently sized data samples: { $k_1, k_2, ...,k_m$ }. The variants {20, 40, 60, 80, 100} or {40, 60, 80, 100} have been found effective. Finally, merge the resulting models using Multi-SLERP from [MergeKit](https://github.com/arcee-ai/mergekit).

 # How to Evaluate the General Text Embedding on MTEB?
 Update the model path in MTEB_test/bash.sh. And then, 
 
 ```
 cd MTEB_test
 sh bash.sh
```

**Note on Model Checkpoint:** Currently, during the anonymous review period, our trained models are hosted on a non-anonymous platform. Our models will be released upon acceptance of this paper.


