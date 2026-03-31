# Text Summarization using FLAN-T5

---

## 📌 Problem Statement

The objective of this assignment is to build a system that generates concise summaries from longer text using modern AI techniques. The goal is to understand how pretrained transformer models can be used for real-world natural language processing tasks.

---

## ⚙️ Approach

* Used the Hugging Face Transformers library to access pretrained models
* Initially attempted summarization using a BART-based pipeline
* Due to issues in output quality and environment compatibility, switched to an instruction-based approach using FLAN-T5
* Implemented summarization using prompt-based instructions
* Designed and tested multiple prompts to observe how output changes
* Controlled output using parameters like `max_length`, `min_length`, and `do_sample`

---

## 🧠 Model Explanation

FLAN-T5 is a transformer-based model that has been **instruction-tuned**, meaning it is trained to follow natural language instructions like:

* “Summarize this text”
* “Translate this sentence”
* “Answer this question”

Unlike traditional models that rely on fixed pipelines, FLAN-T5 performs tasks based on how the input prompt is structured.

### Key Concepts:

* **Transformer Architecture**: Uses attention mechanisms to understand relationships between words
* **Pretrained Model**: Already trained on large datasets, no need for training from scratch
* **Instruction Tuning**: Enables flexible task execution using prompts

---

## ⚠️ Difficulties Faced

### 1. BART Model Pipeline Issues

* Initially used a summarization pipeline with BART
* The output was not a proper summary; it mostly truncated or repeated parts of the input
* In some cases, the pipeline did not behave as expected due to environment/version inconsistencies

### 2. Generic / Incorrect Outputs (FLAN-T5)

* Early outputs were too short or overly generic
* Model sometimes added information not present in the input (hallucination)

### 3. Prompt Sensitivity

* Small changes in prompt wording caused significant variation in output

---

## 🛠️ Resolutions

### 1. Switching from BART to FLAN-T5

* Instead of relying on the default summarization pipeline, switched to:

  ```python
  pipeline("text2text-generation", model="google/flan-t5-base")
  ```
* FLAN-T5 provided better control via prompts and more consistent behavior in the given environment

### 2. Prompt Engineering

* Improved prompt from:

  * “Summarize the following text”
* To:

  * “Summarize the following text accurately without adding new information”

This reduced hallucination and improved factual correctness.

### 3. Parameter Tuning

* Adjusted:

  * `max_length` → control summary size
  * `min_length` → ensure meaningful output
  * `do_sample=False` → make output deterministic and stable

---

## 📊 Results

* Successfully generated concise and meaningful summaries from longer text
* Improved output quality using better prompt design
* Demonstrated how model behavior changes with different prompts

### Example Output:

**Input:**
Artificial Intelligence is rapidly transforming industries across healthcare, finance, and retail...

**Output:**
Artificial Intelligence is transforming industries while raising concerns about job displacement and ethics.

---

## 🧠 Key Learnings

* Model selection significantly affects performance
* Prompt engineering is critical for instruction-tuned models
* Pretrained models can be effectively used without training
* Debugging environment and model behavior is an essential skill in AI development
* Output quality depends on:

  * input length
  * prompt clarity
  * generation parameters

---

## 🚧 Limitations

* Some important details may be lost during summarization
* Output can still be slightly generic in some cases
* Performance depends heavily on prompt quality

---

## ▶️ How to Run

### Step 1 — Install Dependencies

```bash id="u5txdy"
pip install transformers sentencepiece
```

---

### Step 2 — Run the Code

Use Google Colab or a local Python environment.

---

### Step 3 — Example Code

```python id="e7c3jh"
from transformers import pipeline

summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

text = """Your input text here"""

prompt = "Summarize the following text accurately without adding new information:\n" + text

result = summarizer(prompt, max_length=40, min_length=15, do_sample=False)

print(result[0]['generated_text'])
```

---

## ✅ Conclusion

This project demonstrates how transformer-based models can be used for summarization without training from scratch. It also highlights the importance of model selection and prompt engineering in achieving accurate and reliable results.

---
