# agentCLS

This project is for the conference paper "Small Language Models in the Real World: Insights from Industrial Text Classification".

<div style="text-align: center;">
    <img src="photo.png" alt="Figure 1: Model Overview" width="50%">
</div>

## Abstract

With the emergence of ChatGPT, Transformer models have significantly advanced text classification and related tasks. Decoder-only models such as Llama exhibit strong performance and flexibility, yet they suffer from inefficiency on inference due to token-by-token generation, and their effectiveness in text classification tasks heavily depends on prompt quality. Moreover, their substantial GPU resource requirements often limit widespread adoption. Thus, the question of whether smaller language models are capable of effectively handling text classification tasks emerges as a topic of significant interest. However, the selection of appropriate models and methodologies remains largely underexplored. In this paper, we conduct a comprehensive evaluation of prompt engineering and supervised fine-tuning methods for Transformer-based text classification. Specifically, we focus on practical industrial scenarios, including email classification, legal document categorization, and the classification of extremely long academic texts. We examine the strengths and limitations of smaller models, with particular attention to both their performance and their efficiency in video random-access memory (VRAM) utilization, thereby providing valuable insights for the local deployment and application of compact models in industrial settings.

## Methods

In this study, we primarily investigate the effectiveness of **Supervised Fine-Tuning (SFT)** and **Prompt Engineering (PE)** for long-text classification tasks across three distinct datasets, including one derived from a real-world industrial database. Our evaluation encompasses a range of prompting strategies, including:

- **Base Prompting**
- **Few-shot Prompting**
- **Chain-of-Thought (CoT)**
- **Self-consistency CoT**
- **Chain-of-Draft**

as well as advanced fine-tuning methods such as:

- **Soft Prompt Tuning (SPT)**
- **Prefix Tuning (PT)**

## Code Explanation


The `script` directory contains all code necessary for training various models:

- **FT_bert**: Fine-tuning of the Modern BERT model.  
- **FT_llama**: Fine-tuning of the LLaMA model.  
- **SPT**: Includes all Soft Prompt Tuning methods, encompassing both Prefix Tuning and Soft Prompt Tuning approaches.  

These scripts serve as a reference for verifying code consistency and provide a structured template for further research into the impact of fine-tuning techniques on classification tasks. Each folder contains a notebook alongside corresponding `.py` files for testing and execution.

### Prompt Evaluation

Within the `prompt_eval` directory:

- **prompt_testing_gpt.py**: Evaluates prompts using the GPT API.  
- **prompt_testing.py**: Facilitates model evaluation through the vLLM service API.  

### Utility Scripts

The `utils/prompts.py` file includes all prompt designs along with their specific content.


## Examples Of datasets

### LDD - A dataset mainly for classify academic paper topic



**Label: cs.AI**


> **Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization**  
> *Daniel Golovin*  
> **ANTISPAM . GOLOVIN @ GMAIL . COM**  
> California Institute of Technology  
> Pasadena, CA 91125, USA  
>
> *arXiv:1003.3967v5 [cs.LG] 6 Dec 2017*  
>
> *Andreas Krause*  
> **ANTISPAM . KRAUSEA @ ETHZ . CH**  
> ETH Zurich  
> 8092 Zurich, Switzerland  
>
> **Abstract**  
> Many problems in artificial intelligence require adaptively making a sequence of decisions with uncertain outcomes under partial observability. Solving such stochastic optimization problems is a fundamental but notoriously difficult challenge. In this paper, we introduce the concept of adaptive submodularity, generalizing submodular set functions to adaptive policies. We prove that if a problem satisfies this property, a simple adaptive greedy algorithm is guaranteed to be competitive with the optimal policy. In addition to providing performance guarantees for both stochastic maximization and coverage, adaptive submodularity can be exploited to drastically speed up the greedy algorithm by using lazy evaluations. We illustrate the usefulness of the concept by giving several examples of adaptive submodular objectives arising in diverse AI applications including management of sensing resources, viral marketing and active learning. Proving adaptive submodularity for these problems allows us to recover existing results in these applications as special cases, improve approximation guarantees and handle natural generalizations.  
>
> **Keywords:** Adaptive Optimization, Stochastic Optimization, Submodularity, Partial Observability, Active Learning, Optimal Decision Trees  
>
> **1. Introduction**  
> In many problems arising in artificial intelligence one needs to adaptively make a sequence of decisions, taking into account observations about the outcomes of past decisions. Often these outcomes are uncertain, and one may only know a probability distribution over them. Finding optimal policies for decision making in such partially observable stochastic optimization problems is notoriously intractable (see, e.g., Littman et al. (1998)). A fundamental challenge is to identify classes of planning problems for which simple solutions obtain (near-) optimal performance.  
> In this paper, we introduce the concept of adaptive submodularity, and prove that if a partially observable stochastic optimization problem satisfies this property, a simple adaptive greedy algorithm is guaranteed to obtain near-optimal solutions. In fact, under reasonable complexity-theoretic assumptions, no polynomial time algorithm is able to obtain better solutions in general. Adaptive submodularity generalizes the classical notion of submodularity¹, which has been successfully used to develop approximation algorithms for a variety of non-adaptive optimization problems. Submodularity, informally, is an intuitive notion of diminishing returns, which states that adding an element to a small set helps more than adding that same element to a larger (super-) set. A celebrated result of the work of Nemhauser et al. (1978) guarantees that for such submodular functions, a simple greedy algorithm, which adds the element that maximally increases the objective value, selects a near-optimal set of k elements. Similarly, it is guaranteed to find a set of near-minimal cost that achieves a desired quota of utility (Wolsey, 1982), using near-minimum average time to do so (Streeter and Golovin, 0. This work appeared in the Journal of Artificial Intelligence Research (Golovin and Krause, 2011a), and an earlier extended abstract appeared in the International Conference on Learning Theory (Golovin and Krause, 2010).  
> ¹For an extensive treatment of submodularity, see the books of Fujishige (2005) and Schrijver (2003).  
>
> © 2012 Daniel Golovin and Andreas Krause. 
> ...
> 
> ...
> 
> ...
> 
> ...
> 
> ...
> 
> ... 


### EUR - A dataset mainly for classify topic on law topics 

**Label: Regulation**


> **Council Regulation (EC) No 1400/1999 of 24 June 1999**  
> *Fixing the target price for milk and the intervention prices for butter and skimmed-milk powder for the 1999/2000 milk marketing year*  
>
> **COUNCIL REGULATION (EC) No 1400/1999**  
> *of 24 June 1999*  
> *Fixing the target price for milk and the intervention prices for butter and skimmed-milk powder for the 1999/2000 milk marketing year*  
>
> **THE COUNCIL OF THE EUROPEAN UNION**,  
>
> **Having regard to** the Treaty establishing the European Community,  
> **Having regard to** Council Regulation (EEC) No 804/68 of 27 June 1968 on the common organisation of the market in milk and milk products(1), and in particular Articles 3(4) and 5 thereof,  
> **Having regard to** the proposal from the Commission(2),  
> **Having regard to** the opinion of the European Parliament(3),  
> **Having regard to** the opinion of the Economic and Social Committee(4),  
>
> **Whereas:**  
> (1) When fixing the common agricultural prices each year, account should be taken of the objectives of the common agricultural policy; whereas the objectives of the common agricultural policy are in particular to secure a fair standard of living for the agricultural community and to ensure that supplies are available and that they reach the consumers at reasonable prices;  
> (2) The target price for milk should bear a balanced relationship to the prices for other agricultural products and in particular to the prices for beef and veal, and be consistent with the desired general pattern of cattle farming; whereas it is also necessary, in fixing that price, to take account of the Community's efforts to establish a long-term balance between supply and demand on the milk market, allowing for external trade in milk and milk products;  
> (3) The intervention prices for butter and for skimmed-milk powder are intended to contribute to the achievement of the target price for milk; whereas it is necessary to determine price levels in the light of the overall supply and demand situation on the Community market in milk and the opportunities for disposal of butter and skimmed-milk powder on the Community and world markets.  
>
> **For the 1999/2000 milk marketing year, the target price for milk and the intervention prices for milk products shall be as follows:**  
>
> > **TABLE** *(not shown here, please insert if available)*  
>
> **This Regulation shall enter into force on the day of its publication in the Official Journal of the European Communities.**  
>
> **This Regulation shall be binding in its entirety and directly applicable in all Member States.**
> ...
> 
> ...
> 
> ...
> 
> ...
> 
> ...
> 
> ...


### Industrial Email dataset - A industrial dataset mainly for classify emails 

**Label: A reminder**

> **Sent: mercredi 6 octobre 2021 21:49:23**  
> **To:** **** <@.>  
> **Cc:** **** <@.>  
> **Subject:** Re:
>
> Soyez vigilant.e !  
> Ce message est envoyé par un correspondant externe.  
> Veuillez ne pas cliquer sur des liens ou des pièces jointes sauf si ce message est sollicité et que vous avez acquis l'assurance qu'il provient d'une source sûre.  
> Un doute sur son origine ? Prévenez la hotline.  
>
> Bonjour,  
> N'ayant toujours pas obtenu de remboursement, je me permets de relancer la demande annexe.  
>
> Cordialement  
>  
>
> Le message et toutes les pièces jointes (ci-après le "message") sont établis à l’intention exclusive de ses destinataires et sont confidentiels.  
> Si vous recevez ce message par erreur, merci de le détruire et d’en avertir immédiatement l’expéditeur.  
>
> Bonjour,  
> Veuillez trouver en annexe, la facture finale.  
>
> Cordialement  

> Ce message et toutes les pièces jointes (ci-après le "message") sont établis à l’intention exclusive de ses destinataires et sont confidentiels.  
> Si vous recevez ce message par erreur, merci de le détruire et d’en avertir immédiatement l’expéditeur.  
> ...
> 
> ...
> 
> ...
> 
> ...
> 
> ...
> 
> ...


