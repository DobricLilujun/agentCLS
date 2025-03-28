# agentCLS

This project is for the conference paper "Small Language Models in the Real World: Insights from Industrial Text Classification".


## Abstract

With the emergence of ChatGPT, Transformer models have significantly advanced text classification and related tasks. Decoder-only models such as Llama exhibit strong performance and flexibility, yet they suffer from inefficiency on inference due to token-by-token generation, and their effectiveness in text classification tasks heavily depends on prompt quality. Moreover, their substantial GPU resource requirements often limit widespread adoption. Thus, the question of whether smaller language models are capable of effectively handling text classification tasks emerges as a topic of significant interest. However, the selection of appropriate models and methodologies remains largely underexplored. In this paper, we conduct a comprehensive evaluation of prompt engineering and supervised fine-tuning methods for Transformer-based text classification. Specifically, we focus on practical industrial scenarios, including email classification, legal document categorization, and the classification of extremely long academic texts. We examine the strengths and limitations of smaller models, with particular attention to both their performance and their efficiency in video random-access memory (VRAM) utilization, thereby providing valuable insights for the local deployment and application of compact models in industrial settings.

## Methods

We mainly consider ***Soft Prompt Tuning (SPT)***, ***Supervised Fine Tuning (SFT)*** and ***Prompt Engineering (PE)*** to investigate the classification performance on three different datasets (One from the real industry database).

