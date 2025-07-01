i. GCN+BERT Relation Extraction Model:

This repository contains the **GCN+BERT Relation Extraction model**, designed to classify **semantic relations** between entities in text. The model integrates **BERT embeddings** with **Graph Convolutional Networks (GCNs)** to leverage both **contextual and syntactic information**, making it highly effective for **relation classification tasks**, particularly on the **SemEval-2010 Task 8 dataset**.


ii. 📂 Repository Structure


Approach B/
│
├── Code/
    ├── gcn_bert.py             # Main script for training and inference of the GCN+BERT model
    ├── evaluation_method.py    # Script for evaluating model performance and analyzing errors
    ├── inference_model.py      # Script for real-time inference with sample input provided

├── Results
    ├── train_confusion_matrix.png  # Visualization of training set misclassifications
    ├── val_confusion_matrix.png    # Visualization of validation set misclassifications
    ├── test_confusion_matrix.png   # Visualization of test set misclassifications
    ├── train_error_analysis.csv    # Detailed misclassification report for training set
    ├── val_error_analysis.csv      # Detailed misclassification report for validation set
    ├── test_error_analysis.csv     # Detailed misclassification report for test set

└── README.md               # Documentation and usage instructions


---

## **Dependencies & Installation**

Ensure you have the following dependencies installed:

```bash
# Install required Python packages
pip install torch torchvision torchaudio
pip install transformers spacy networkx scikit-learn pandas numpy tqdm matplotlib
```

Download and install the **spaCy** English model for **dependency parsing**:

```bash
python -m spacy download en_core_web_sm
```

---

## **Dataset (SemEval-2010 Task 8)**

The model is trained and evaluated on the **SemEval-2010 Task 8 dataset**, which contains **10,717 sentences** annotated with **19 relation types**. The dataset is available on **Hugging Face Datasets**:

🔗 [SemEval-2010 Task 8 Dataset](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8)

We would recommend using the .parquet files and augmented dataset as we have written the code based on these files which is  made available in the drive link.
The google drive link can be accessed with the below link:
https://drive.google.com/drive/folders/19_OWDi2RpDH4afdsnDj1tNMqXdBHrVYp?usp=sharing

Google Drive Structure under Approach_B:

├──Dataset
    ├── test-00000-of-00001.parquet   # Test dataset for the model
    ├── train-00000-of-00001.parquet  # Training dataset for the model
├── sr_augmented_data
    ├──augmented_dataset.arrow
        ├── data-00000-of-00001.arrow
        ├── dataset_info.json
        ├── state.json

├── best_model_final.pt         # Saved best model for inference

---

## **Running the Model**

### **Training the Model**

To train the **GCN+BERT model** on the **SemEval-2010 Task 8 dataset**, run:

```bash
python gcn_bert.py
```
---

## **Evaluating the Model**

To evaluate the trained model on test data, run:

```bash
python evaluation_method.py
```


---

## **Inference - Predicting Relations in New Sentences**

To run the inference on test/input data, run:

```bash
python inference_model.py
```
To classify the **semantic relation** between two entities in a sentence, use the `predict_relation_from_text()` function.

### **Usage Example**

```python
test_sentence = "The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
predicted_relation, confidence = predict_relation_from_text(model, tokenizer, test_sentence, device)
print(f"\nPredicted relation: {predicted_relation} with confidence: {confidence:.4f}")
```

### **Input Format**
- The sentence must include **two explicitly marked entities** using the following format:  
  - `<e1> entity_1 </e1>`  
  - `<e2> entity_2 </e2>`  

#### **Example Input:**
```plaintext
"The liquid was stored in a <e1>bottle</e1> inside a <e2>container</e2>."
```

### **Output Format**
- **Predicted Relation (Integer ID):** The assigned relation class.
- **Confidence Score:** The **softmax probability** indicating the model’s certainty in the classification.

#### **Example Output:**
```bash
Predicted relation: 3 with confidence: 0.9207
```
---

## **Model Performance**

| Dataset         | Accuracy (%) | Macro-F1 Score (%) | Error Rate (%) |
|----------------|-------------|--------------------|----------------|
| **Training Set**   | 96.78       | 91.29              | 3.22           |
| **Validation Set** | 97.19       | 91.95              | 2.81           |
| **Test Set**       | 83.29       | 79.21              | 16.71          |

The model achieves **high accuracy and F1-score** in **training and validation**, but experiences a **performance drop on the test set** due to **semantic overlap and entity ambiguity**.

---

## **Comparison with AGGCN (Base Paper)**

| Metric             | **GCN+BERT (Our Model)** | **AGGCN (Base Paper)** |
|--------------------|------------------------|------------------------|
| **Accuracy (%)**      | 83.29                   | 81.80                   |
| **Macro-F1 Score (%)** | 79.21                   | 77.50                   |
| **Error Rate (%)**     | 16.71                   | 18.20                   |

### **Key Takeaways:**
✅ **GCN+BERT improves classification accuracy** over AGGCN by leveraging **BERT embeddings** instead of **static GloVe embeddings**.  
✅ **Lower error rate** indicates **better generalization** to unseen data.  
✅ **Preserving full dependency graphs** improves **syntactic feature propagation**.

---

## **Key Resources**
- [SemEval-2010 Task 8 Dataset](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8)
- [BERT Model (Hugging Face)](https://huggingface.co/bert-base-uncased)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [spaCy Dependency Parsing](https://spacy.io/)

---

## **Future Work**
- **Incorporating Graph Attention Networks (GATs)** to better weigh important dependency edges.
- **Using contrastive learning techniques** to refine closely related relation types.
- **Expanding dataset diversity** to improve generalization on **low-frequency relations**.

---

## **Generative AI disclosure**
- OpenAI’s ChatGPT and Deepseek were utilized in a limited manner, exclusively for enhancing code clarity, including naming conventions and structuring of functions, and assistance in debugging. 
- The core methodology, ideas, logic, and reasoning presented in this work were developed entirely independently by the authors.