
prompt_EUR_BASE= """Return the classification answer after a separator ####. Do not return any preamble, explanation, or reasoning.

Classify the **input** text into one of the following categories based on the descriptions provided, and explicitly provide the output classification at the end. 

Categories:
1. **Decision** - Choose this category if the text involves making a choice or selecting an option.
2. **Directive** - Use this category if the text instructs or commands an action.
3. **Regulation** - Appropriate for texts that stipulate rules or guidelines.

<<<START OF INPUT>>>

{input}

<<<END OF INPUT>>>
"""

prompt_EUR_COT = """Think step by step to answer the following question. Return the classification answer at the end of response after a separator ####.

Classify the **input** text into one of the following categories based on the descriptions provided, and explicitly provide the output classification at the end. 

Categories:
1. **Decision** - Choose this category if the text involves making a choice or selecting an option.
2. **Directive** - Use this category if the text instructs or commands an action.
3. **Regulation** - Appropriate for texts that stipulate rules or guidelines.

<<<START OF INPUT>>>

{input}

<<<END OF INPUT>>>
"""

prompt_EUR_COD = """Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.

Classify the **input** text into one of the following categories based on the descriptions provided, and explicitly provide the output classification at the end. 

Categories:
1. **Decision** - Choose this category if the text involves making a choice or selecting an option.
2. **Directive** - Use this category if the text instructs or commands an action.
3. **Regulation** - Appropriate for texts that stipulate rules or guidelines.

<<<START OF INPUT>>>

{input}

<<<END OF INPUT>>>
"""



prompt_EUR_FEW_SHOT = """Return the classification answer after a separator ####. Do not return any preamble, explanation, or reasoning.

Classify the **input** text into one of the following categories based on the descriptions and examples provided, and explicitly provide the output classification at the end. 

Categories:
1. **Decision** - Choose this category if the text involves making a choice or selecting an option.
2. **Directive** - Use this category if the text instructs or commands an action.
3. **Regulation** - Appropriate for texts that stipulate rules or guidelines.


## Examples for Classification

### Example: 
**Input Text:**  
"Commission Regulation (EEC) No 3644/88, enacted on 23 November 1988, established the minimum price for blood oranges withdrawn from the market and sold to processing industries for the 1988/89 marketing year. This regulation, rooted in the Treaty establishing the European Economic Community and Council Regulation (EEC) No 1035/72, specifies that the minimum selling price should be set prior to each marketing year, considering the normal supply price to the industry. For the specified marketing year, the price was fixed at 52.42 ECU per tonne net, ex warehouse. The regulation, which aligns with the advice of the Management Committee for Fruit and Vegetables, became effective on 1 December 1988 and is legally binding and directly applicable across all Member States. This measure ensures a standardized price for blood oranges destined for processing, reflecting broader market management strategies within the European Economic Community's framework for fruit and vegetables."
#### 
Regulation


### Example: 
**Input Text:**  
"Directive 2003/87/EC established a scheme for greenhouse gas emission allowance trading within the Community, in line with the Kyoto Protocol's objectives. This Directive lays down measures aimed at reducing greenhouse gas emissions significantly to combat climate change. It provides for a cap on the total amount of certain greenhouse gases that can be emitted by installations covered by the scheme and allows for the trading of emission allowances. This system aims to reduce emissions economically and effectively, ensuring that the Community can meet its Kyoto targets."
#### 
Directive

### Example: 
**Input Text:**  
"Council Decision 2007/621/EC, Euratom appoints Ms. Mette Kindberg as a Danish member of the European Economic and Social Committee, replacing Ms. Randi Iversen, who resigned. Her appointment lasts for the remainder of the term until September 20, 2010. The decision takes effect upon adoption."  
**Classification Reasoning:**  
This text describes an official decision made by the European Council regarding the appointment of a specific individual to a committee. It specifies the replacement of a member, the duration of the appointment, and the decision's immediate effect upon adoption. Since it involves a concrete administrative action rather than broad legislative regulations, it fits within the "Decision" category.  
#### 
Decision 

<<<START OF INPUT>>>

{input}

<<<END OF INPUT>>>
"""

prompt_SELF_CONSIS = """Return the classification answer after a separator ####. Do not return any preamble, explanation, or reasoning. 

You will be provided three thinking paths for answering the text classification question, and the conclusions from the three paths will be compared. If two or more paths arrive at the same classification result, that will be selected as the most consistent answer; if all three paths differ, answer with the most plausible classification based on the overall reasoning.

Question: 
{question_prompt}

Path 1:
{path_1}

Path 2:
{path_2}

Path 3:
{path_3}

Return the classification answer after a separator ####. Do not return any preamble, explanation, or reasoning. 
"""


prompt_LDD_BASE = """Return the classification answer after a separator ####. Do not return any preamble, explanation, or reasoning.

Classify the **input** text into one of the following categories based on the descriptions provided, and explicitly provide the output classification at the end. 

Categories:
- **cs.AI**: Involves topics related to Artificial Intelligence.
- **cs.CE**: Related to Computational Engineering.
- **cs.CV**: Pertains to Computer Vision.
- **cs.DS**: Concerns Data Structures.
- **cs.IT**: Deals with Information Theory.
- **cs.NE**: Focuses on Neural and Evolutionary Computing.
- **cs.PL**: Involves Programming Languages.
- **cs.SY**: Related to Systems and Control.
- **math.AC**: Pertains to Commutative Algebra.
- **math.GR**: Involves Group Theory.
- **math.ST**: Related to Statistics Theory.

<<<START OF INPUT>>>

{input}

<<<END OF INPUT>>>
"""

prompt_LDD_COT = """Think step by step to answer the following question. Return the classification answer at the end of response after a separator ####.

Classify the **input** text into one of the following categories based on the descriptions provided, and explicitly provide the output classification at the end. 

Categories:
- **cs.AI**: Involves topics related to Artificial Intelligence.
- **cs.CE**: Related to Computational Engineering.
- **cs.CV**: Pertains to Computer Vision.
- **cs.DS**: Concerns Data Structures.
- **cs.IT**: Deals with Information Theory.
- **cs.NE**: Focuses on Neural and Evolutionary Computing.
- **cs.PL**: Involves Programming Languages.
- **cs.SY**: Related to Systems and Control.
- **math.AC**: Pertains to Commutative Algebra.
- **math.GR**: Involves Group Theory.
- **math.ST**: Related to Statistics Theory.

<<<START OF INPUT>>>

{input}

<<<END OF INPUT>>>
"""

prompt_LDD_COD = """Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.
Classify the **input** text into one of the following categories based on the descriptions provided, and explicitly provide the output classification at the end. 

Categories:
- **cs.AI**: Involves topics related to Artificial Intelligence.
- **cs.CE**: Related to Computational Engineering.
- **cs.CV**: Pertains to Computer Vision.
- **cs.DS**: Concerns Data Structures.
- **cs.IT**: Deals with Information Theory.
- **cs.NE**: Focuses on Neural and Evolutionary Computing.
- **cs.PL**: Involves Programming Languages.
- **cs.SY**: Related to Systems and Control.
- **math.AC**: Pertains to Commutative Algebra.
- **math.GR**: Involves Group Theory.
- **math.ST**: Related to Statistics Theory.

<<<START OF INPUT>>>

{input}

<<<END OF INPUT>>>
"""


prompt_LDD_FEW_SHOT = """Return the classification answer after a separator ####. Do not return any preamble, explanation, or reasoning.

Classify the **input** text into one of the following categories based on the descriptions provided, and explicitly provide the output classification at the end. 

Categories:
- **cs.AI**: Involves topics related to Artificial Intelligence.
- **cs.CE**: Related to Computational Engineering.
- **cs.CV**: Pertains to Computer Vision.
- **cs.DS**: Concerns Data Structures.
- **cs.IT**: Deals with Information Theory.
- **cs.NE**: Focuses on Neural and Evolutionary Computing.
- **cs.PL**: Involves Programming Languages.
- **cs.SY**: Related to Systems and Control.
- **math.AC**: Pertains to Commutative Algebra.
- **math.GR**: Involves Group Theory.
- **math.ST**: Related to Statistics Theory.

## Examples for Classification:

### Example: 
**Input Text:** 
This paper introduces a novel algorithm for optimizing neural network architecture using evolutionary strategies.
####
cs.AI

### Example: 
**Input Text:** 
A study on the computational modeling of fluid dynamics in complex engineering systems.
####
cs.CE

### Example: 
**Input Text:** 
Enhancements in facial recognition algorithms through deep learning techniques.
####
cs.CV

### Example: 
**Input Text:**
Optimizing search algorithms for faster retrieval in database management systems.
####
cs.DS

### Example: 
**Input Text:**
Exploring new methods for data compression in high-speed networks.
####
cs.IT

### Example: 
**Input Text:**
The impact of genetic algorithms on solving complex optimization problems.
####
cs.NE

### Example: 
**Input Text:**
Developing a new functional programming language designed for concurrent computing.
####
cs.PL

### Example: 
**Input Text:**
Innovative control strategies for autonomous drone navigation.
####
cs.SY

### Example: 
**Input Text:**
Research on the application of Noetherian rings in solving algebraic geometry problems.
####
math.AC

### Example: 
**Input Text:**
Analyzing the symmetrical properties of finite groups and their applications in cryptography.
####
math.GR

### Example: 
**Input Text:**
Using Bayesian networks to predict stock market trends based on historical data.
####
math.ST

<<<START OF INPUT>>>

{input}

<<<END OF INPUT>>>
"""


prompt_IE_BASE = """Return the classification answer after a separator ####. Do not return any preamble, explanation, or reasoning.

Classify the latest email in the provided email history into one of the following categories based on the descriptions, and explicitly provide the output classification at the end.

Categories:
- **Reminder**: Choose this category if the latest email references a past request, follow-up, or attempts to prompt the recipient about an earlier communication.
- **Not Reminder**: Use this category if the latest email introduces a new topic, does not reference past communication, or does not act as a follow-up.

<<<START OF EMAIL HISTORY>>>

{input}

<<<END OF EMAIL HISTORY>>>"""


prompt_IE_COT = """Think step by step to answer the following question. Return the classification answer at the end of response after a separator ####.

Classify the latest email in the provided email history into one of the following categories based on the descriptions, and explicitly provide the output classification at the end.

Categories:
- **Reminder**: Choose this category if the latest email references a past request, follow-up, or attempts to prompt the recipient about an earlier communication.
- **Not Reminder**: Use this category if the latest email introduces a new topic, does not reference past communication, or does not act as a follow-up.

<<<START OF EMAIL HISTORY>>>

{input}

<<<END OF EMAIL HISTORY>>>
"""

prompt_IE_COD = """Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.

Classify the latest email in the provided email history into one of the following categories based on the descriptions, and explicitly provide the output classification at the end.

Categories:
- **Reminder**: Choose this category if the latest email references a past request, follow-up, or attempts to prompt the recipient about an earlier communication.
- **Not Reminder**: Use this category if the latest email introduces a new topic, does not reference past communication, or does not act as a follow-up.

<<<START OF EMAIL HISTORY>>>

{input}

<<<END OF EMAIL HISTORY>>>
"""


prompt_IE_FEW_SHOT = """Return the classification answer after a separator ####. Do not return any preamble, explanation, or reasoning.
Classify the latest email in the provided email history into one of the following categories based on the descriptions and examples provided, and explicitly provide the output classification at the end.

Categories:
- **Reminder**: Choose this category if the latest email references a past request, follow-up, or attempts to prompt the recipient about an earlier communication.
- **Not Reminder**: Use this category if the latest email introduces a new topic, does not reference past communication, or does not act as a follow-up.

## Examples for Classification

### Example: 
**Input Email History:**  
For some time now, we have not heard from you. Could you inform us by 05/02/2024 of the current status of this file? 
#### 
Reminder


### Example: 
**Input Email History:**  
We allow ourselves to return to this matter. We are waiting for the confirmation of the guarantees and responsibility of our principal. We are following up with our principal.
#### 
Reminder

<<<START OF EMAIL HISTORY>>>

{input}

<<<END OF EMAIL HISTORY>>>
"""


# This is a META prompt template from OpenAI
META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt in JSON format to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - Your output should adhere to the following JSON Example below. Do not include any additional commentary. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")


# Task Descriptions
{instructions}

# Additional Information
{additional_details}

# Previous Prompt
{previous_prompt}

# JSON Example
{examples}

""".strip()


META_PROMPT_JSON_FREE = """
Given a task description or existing prompt, produce a detailed few-shot prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)

# Task Descriptions
{instructions}

# Additional Information
{additional_details}

# Previous Prompt
{previous_prompt}


""".strip()


prompt_template_description_dict = {
    "instructions": ("Your task is to update a well-structured provided prompt for a text classification task based on category descriptions. "
                     "Ensure clarity and precision in the prompt so the language model can accurately classify the input. "
                     "Incorporate category descriptions to guide the classification process effectively. "
                     "Maintain a structured format to enhance understanding and consistency. Keep original '{input}' string at the end in the generated prompt."),
    "additional_details": "\nCategory Descriptions:\n {labels_description}",
    "output_format": "The output should be a JSON format with the updated prompt and the refinement reason.",
    "examples": """{"updated_prompt": "content of updated prompt", "reason": "reason for the refinement"}
"""
}

prompt_template_description = (
    "Your task is to generate a well-structured prompt for a text classification task based on category descriptions.\n\n"
    
    "### Instructions:\n"
    "1. **Ensure clarity and precision** in the prompt so the language model can accurately classify the input.\n"
    "2. **Incorporate category descriptions** to guide the classification process effectively.\n"
    "3. **Maintain a structured format** to enhance understanding and consistency.\n\n"
    
    "**Category Descriptions:**\n"
    "{labels_description}\n\n"
    
    "**Please provide only the updated prompt with these refinements. Do not include any other content.**"
)

 

prompt_template_COT_dict = {
    "instructions": ("Your task is to update a well-structured provided prompt to a chain of reasoning prompt for a text classification task based on category descriptions. "
                     "Ensure clarity and precision in the prompt so the language model can accurately classify the input. "
                     "Incorporate category descriptions to guide the classification process effectively. "
                     "Maintain a structured format to enhance understanding and consistency. Keep original '{input}' string at the end in the generated prompt."),
    "additional_details": "\nCategory Descriptions:\n {labels_description}",
    "output_format": "The output should be a JSON format with the updated prompt and the refinement reason.",
    "examples": """{"updated_prompt": "content of updated prompt", "reason": "reason for the refinement"}"""
}


prompt_template_few_shot_dict = {
  "instructions": "Your task is to **iteratively refine** the given prompt into a **few-shot prompt** for a text classification task. The goal is to construct a structured prompt that includes relevant **few-shot examples**, helping the model generalize classification decisions effectively.\n\n"
                  "Follow these steps:\n"
                  "1. **Analyze the category descriptions and input examples**: Extract key details that define each category and ensure the selected examples align with these definitions.\n"
                  "2. **Progressively add examples**: Integrate one example at a time into the few-shot prompt, ensuring they are diverse and representative of the classification task.\n"
                  "3. **Maintain clarity and relevance**: Keep the examples concise and well-structured to avoid ambiguity. Each addition should enhance the model’s ability to classify accurately.\n"
                  "4. **Preserve the original '{input}' placeholder**: The final prompt must retain this placeholder, ensuring it is the part where new classification inputs will be inserted.\n\n"
                  "Repeat this refinement process until the prompt is **clear, structured, and contains sufficient few-shot examples** to effectively guide classification.",
  
  "additional_details": "\nCategory Descriptions and Additional Examples:\n{labels_description}",

  "output_format": "The output must only contain the fully refined **few-shot prompt**, formatted correctly for immediate use in classification tasks. No explanations or extra text should be included.",

  "examples": "{\"updated_prompt\": \"Here is an example of a structured few-shot prompt: \\n\\n"
              "Example 1: \\n"
              "- Input: <example_text_1> \\n"
              "- Classification: <category_1> \\n\\n"
              "Example 2: \\n"
              "- Input: <example_text_2> \\n"
              "- Classification: <category_2> \\n\\n"
              "{input}\", \"reason\": \"This refinement improves clarity by adding diverse examples that align with category descriptions, ensuring a more structured few-shot prompt.\"}"
}


prompt_template_few_shot = (
    "You are tasked with refining a prompt for a text classification task based on category descriptions. "
    "You will be provided with a previously generated prompt, a new label, and a new text example. "
    "Your goal is to integrate the new information while maintaining a structured reasoning approach.\n\n"
    
    "### Instructions:\n"
    "1. **Update the prompt** by incorporating the new category and text example.\n"
    "2. **Ensure clarity and logical flow** by structuring the classification reasoning in a step-by-step manner.\n"
    "3. **Follow the few-shot structure for prompt** when determining if a text belongs to a category:\n"
    "   - The definition of all categories and its characteristics.\n"
    "   - Example 1 and category 1\n"
    "   - Example 2 and category 2\n"
    "4. **Do not generate any additional explanations, comments, or unrelated content.**\n\n"
    
    "### Example Structure:\n"
    "   - Determine whether the text belongs to *Category X*:\n"
    "     - Condition 1: [Description]\n"
    "     - Condition 2: [Description]\n"
    "     - If both conditions are satisfied, classify the text as *Category X*.\n\n"
    
    "**Previous Prompt:**\n"
    "{previous_prompt}\n\n"
    
    "**New Text and Category:**\n"
    "{labels_string}\n\n"
    
    "**Please provide only the updated prompt with these refinements. Do not include any other content.**"
)

