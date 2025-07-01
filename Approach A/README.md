This repository contains all the files related to the implementation of Approach A 
Leveraging Distant Supervision in Bert Models to enhance Relation Extraction

Contributions
-> Implementation of the novel technique of Distant Supervision using SemEval dataset and training BERT and R-BERT models for Relation Extraction Task
-> Implementation of alternative input representation for entities in the sentence.
-> Rigorous experimentation with these techniques and comination of models, tuning best model, performance evaluation and comparison with different models.

📂 Repository Structure and Details

Approach A/
│
├── Distant_Supervision.ipynb   # Implements dataset augmentation, starting with a knowledge base derived from SemEval dataset
    DS_CW_BERT_model.ipynb     	# Contains model training, validation, and hyperparameter tuning, along with test results.
    Inference_Mode.ipynb       	# notebook for real-time inference using the saved model, with user input format and example provided in the notebook.

├── Experimental Setup
    ├── entity_tags_base_bert_model.ipynb  	# BERT baseline using entity markers ($#) as input format  	
    ├── entitytags_cw_ds_bert.ipynb   		# BERT trained with class weighting (CW) and distant supervision (DS) using entity markers ($#)
    ├── our_input_scheme_base_bert_bodel.ipynb  # BERT baseline with our input scheme 
    ├── R_bert_with_our_dataset.ipynb    	Fine-tunes R-BERT on our distantly supervised dataset

└── README.md               	# Documentation and usage instructions


Implementation Details
-> All training and inference were performed on Google Colab (T4 GPU). This is the recommended environment for running the notebooks.
-> For Inference_Mode, the model needs to be downloaded from the google drive and put into the notebook environment.

Dataset (SemEval-2010 Task 8)

The model is trained and evaluated on the SemEval-2010 Task 8 dataset, which contains a training set of 8,000 sentences and a test set of 2717 sentences.
-> It contains 19 relation types and is available on all the standard libraries such as Hugging Face Datasets etc.
(Official dataset link - https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8 )

-> To ensure the dataset is pre-processed and formatted correctly before training, we recommend using the files given in the google drive link.
-> These are files curated for this repository codes and just contain formatting changes from the official link.

Google drive link: https://drive.google.com/drive/folders/19_OWDi2RpDH4afdsnDj1tNMqXdBHrVYp?usp=sharing
Google Drive Structure under Approach_A:

├── cw_ds_bert_model.pt        	  # Saved model for inference

├──Dataset
    ├── large_ds_df_exported.csv  # Distant Supervised training dataset for the model
    ├── TEST_File.txt 		  # Test dataset for the model
    
    ├── Datasets for Experimental setup
    	├── dataset.csv 	  # distant supervised training file used for R-BERT  model 
        ├── test.tsv		  # Test dataset for the R-BERT model	

Distant Supervised Dataset
-> The augmented dataset which contains the expanded training examples (around 16000) is dervied from Distant_supervision.ipynb 

-> To run training files, the datasets needs to be downloaded from the google drive and put into the notebook environment. 


Model Performance - Distant Supervised Class Weighted BERT

  	             | Accuracy (%)|      Macro-F1 Score (%) 	 | 
				   |(18 classes -Without 'other')| 
|--------------------|-------------|-----------------------------|
| Validation Set     | 89          | 71                 	 | 
| Test Set           | 80          | 66              	 	 | 

Enhancements
-> The evaluation metrics used were accuracy and macro-averaged F1 score without the class ‘Other’, reported as F1(-O). 
-> Distant Supervision significantly improves BERT’s performance, with validation accuracy increasing from 73% to 89% and F1 rising to
71% with tuning. 
-> The test F1 improves by 10% when using our input representation compared to $# markers.
-> R-BERT also benefits from distant supervision, with a F1 improvement from 79% to 80.5%. 
-> The proposed DS-CW-BERT model, taking around 5 minutes of training per epoch, gives a validation accuracy of 89% and a
test accuracy of 80%, while R-BERT took 20+ minutes per epoch, giving a similar test accuracy.

Hyperparamter Tuning
-> Probabilistic grid search was used on the proposed model with the below novel parameters reporting the high accuracy (mentioned above) for our model.

|   Hyperparameter   |  Value      |
|--------------------|-------------|
|    Epochs          |  5          |
|    Batch Size      |  64         |
|    Learning Rate   |  7.6 × 10−5 |
|    Weight Decay    |  1 × 10−8   |
|    Epsilon         |  0.01       |
|    Warmup Steps    |  10%        | 

 
References
-> To gain a deeper understanding of relation classification on the SemEval dataset, we examined relevant open-source codebases, adapting best practices where applicable while ensuring the development of an independent approach tailored to our specific research objectives. Additionally, while surveying relevant work, we reviewed an MSc thesis [1] available on GitHub, which provided insights into prior experimental setups and implementation techniques. 
-> This work is independently developed and these resources have contributed to our understanding of prior work, guiding our research and implementation choices.
-> To evaluate the effectiveness of our distantly supervised dataset, we used an existing R-BERT implementation [2] available on GitHub as a baseline. This allowed for a controlled comparison between a model trained on the original SemEval data and that trained on our expanded dataset. 
-> All dataset modifications, training configurations, and analysis were independently designed as part of this work, while the original implementation provided a foundation for benchmarking.

[1] https://github.com/nb20593/Semeval-2010-task-8-Relational-extraction/blob/main/CE901-Msc%20dissertation.pdf
[2] https://github.com/heraclex12/R-BERT-Relation-Classification

Generative AI disclosure:
-> OpenAI’s ChatGPT and Deepseek were utilized in a limited manner, exclusively for enhancing code clarity, including naming conventions and structuring of functions, and assistance in debugging. 
-> The core methodology, ideas, logic, and reasoning presented in this work were developed entirely independently by the authors.
