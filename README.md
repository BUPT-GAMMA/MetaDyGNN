# MetaDyGNN

Source code for WSDM 2022 paper "Few-shot Link Prediction in Dynamic Networks"


## Environment Settings

`python==3.8.5` 

`torch==1.4.0`

`numpy==1.21.0` 

`scikit_learn==0.24.1`

`pandas==1.2.1` 

GPU: GeForce RTX 2080 Ti

CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz


## Usage

`python code/main.py`

Here, you can change hyperparameter settings in code/Config.py.


## Preprocessing

If you download datasets from urls in the paper, you need to preprocess the data by process.py or process_dblp.py.


## Cite


@inproceedings{yang2022few,

  title={Few-shot Link Prediction in Dynamic Networks},
  
  author={Yang Cheng, Wang Chunchen, Lu Yuanfu, Gong Xumeng, Shi Chuan, Wang Wei and Zhang Xu},
  
  booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  
  pages={1245--1255},
  
  year={2022}
  
}
