## EXAMPLE : train a ruby model

       step1：check training data file and test file in dir /HFERR/data

       step2：Set parameters in the configuration file /HFERR/src/config.py

       step3：Run /HFERR/src/train.py file to start training.

       step4： (1) Run /HFERR/inference/fusion_extract.py to extracting queries and codes features
       
               (2) Run /HFERR/inference/result.py to calculate mrr scores.
                
               * The reason why there are two steps in inference is the limitation of memory size.


## Environmental requirements
        The latest version: python, pytorch, transformer,





## folder structure

1： \data： Trainning data and Test data

       NOTE：（1): Due to the large data set, only a part of the data is left here for quick test code.
       
              (2）: The complete dataset can be downloaded from https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/codesearch.

2： \embedding： During inference, temporarily save feature vectors.

3： \inference ： inference code

       NOTE： (1) fusion_extract.py to extracting queries and codes features for inference. The rest of the code files are called by fusion_extract.py.
       
              (2) result.py to calculate mrr scores.

4： \outputs_model ： save trained model

5： \ src ： training code

       NOTE： （1)run train.py to train a model, the rest of the code files are called by train.py.
       
               (2）config.py is configuration file.



