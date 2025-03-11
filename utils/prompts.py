
prompt_eurlex57k_description_classification = """Classify the given input text into one of the following categories based on the descriptions provided.

Categories:
1. **Decision** - Choose this category if the text involves making a choice or selecting an option.
2. **Directive** - Use this category if the text instructs or commands an action.
3. **Regulation** - Appropriate for texts that stipulate rules or guidelines.

Text for Classification:
{input}

"""

prompt_eurlex57k_COT_classification = """To accurately classify the input text, consider the following category descriptions: 
- **Decision**: Texts that involve making or describing decisions. 
- **Directive**: Texts that instruct or command specific actions. 
- **Regulation**: Texts that outline rules, guidelines, or laws. 

To accurately classify text into the categories of 'Decision', 'Directive', or 'Regulation', follow this structured reasoning process: 
1. Begin by comprehensively reading the text to grasp its content and intent. 
2. Evaluate whether the text primarily involves making a choice or reaching a conclusion, which would classify it as a 'Decision'. Identify critical phrases and structures that suggest a decision-making process. Place this assessment under the reasoning field. 
3. If the text primarily directs or instructs actions, categorize it as a 'Directive'. Look for directive language or commands that are indicative of this classification. This evaluation should also be placed under the reasoning field. 
4. Assess if the text establishes rules or guidelines, categorizing it as a 'Regulation'. Search for terms and formats typical of regulatory texts and include this in the reasoning field. 
5. After completing these reasoning steps, assign the text to the most fitting category based on the evidence gathered. This conclusion should be detailed in the conclusion field. {input}

Please classify the following input text into the appropriate category: 

{input}
"""


prompt_eurlex57k_COT_classification = """To accurately classify the input text, consider the following category descriptions: 
- **Decision**: Texts that involve making or describing decisions. 
- **Directive**: Texts that instruct or command specific actions. 
- **Regulation**: Texts that outline rules, guidelines, or laws. 

To accurately classify text into the categories of 'Decision', 'Directive', or 'Regulation', follow this structured reasoning process: 
1. Begin by comprehensively reading the text to grasp its content and intent. 
2. Evaluate whether the text primarily involves making a choice or reaching a conclusion, which would classify it as a 'Decision'. Identify critical phrases and structures that suggest a decision-making process. Place this assessment under the reasoning field. 
3. If the text primarily directs or instructs actions, categorize it as a 'Directive'. Look for directive language or commands that are indicative of this classification. This evaluation should also be placed under the reasoning field. 
4. Assess if the text establishes rules or guidelines, categorizing it as a 'Regulation'. Search for terms and formats typical of regulatory texts and include this in the reasoning field. 
5. After completing these reasoning steps, assign the text to the most fitting category based on the evidence gathered. This conclusion should be detailed in the conclusion field. {input}

Please classify the following input text into the appropriate category: 

{input}
"""


prompt_eurlex57k_few_shots_classification = """## Task Description
Your task is to classify input texts based on specific categories. Each category has unique characteristics that the text must exhibit to be classified accordingly. The objective is to ensure accurate classification by critically analyzing the content, context, and specific details of the input text.

## Category Descriptions

### Label: Regulation  
**Description:** Texts that detail formal regulations issued by governmental or other authoritative bodies. These texts typically specify rules or laws that are legally binding and directly applicable across relevant jurisdictions. They often include details such as effective dates, the entities affected, and the specific mandates or prohibitions imposed.

### Label: Directive  
**Description:** The "Directive" category is designated for texts that detail formal legislative directives issued by authoritative bodies such as the European Parliament and the Council. These texts typically discuss amendments to existing laws, set new regulatory standards, and outline implementation guidelines and deadlines. The content should reflect the introduction or modification of policies, particularly in the context of environmental, health, or safety regulations.

## Examples for Classification

### Example 1: Regulation
**Input Text:**  
"Commission Regulation (EEC) No 3644/88, enacted on 23 November 1988, established the minimum price for blood oranges withdrawn from the market and sold to processing industries for the 1988/89 marketing year. This regulation, rooted in the Treaty establishing the European Economic Community and Council Regulation (EEC) No 1035/72, specifies that the minimum selling price should be set prior to each marketing year, considering the normal supply price to the industry. For the specified marketing year, the price was fixed at 52.42 ECU per tonne net, ex warehouse. The regulation, which aligns with the advice of the Management Committee for Fruit and Vegetables, became effective on 1 December 1988 and is legally binding and directly applicable across all Member States. This measure ensures a standardized price for blood oranges destined for processing, reflecting broader market management strategies within the European Economic Community's framework for fruit and vegetables."

**Classification Reasoning:**  
This text outlines a specific regulation setting a minimum price for blood oranges, indicating the effective date and the legal framework it operates within, thereby affecting the relevant market practices. It also mentions the direct applicability and binding nature of the rule across Member States, characteristics typical of regulatory texts.

**Assigned Category:** Regulation

### Example 2: Directive
**Input Text:**  
"Directive 2003/87/EC established a scheme for greenhouse gas emission allowance trading within the Community, in line with the Kyoto Protocol's objectives. This Directive lays down measures aimed at reducing greenhouse gas emissions significantly to combat climate change. It provides for a cap on the total amount of certain greenhouse gases that can be emitted by installations covered by the scheme and allows for the trading of emission allowances. This system aims to reduce emissions economically and effectively, ensuring that the Community can meet its Kyoto targets."

**Classification Reasoning:**  
This text describes a legislative directive issued by the European Union that sets regulatory standards for greenhouse gas emissions, aligns with international environmental goals, and outlines a trading scheme for emission allowances. It clearly specifies the policy's purpose, its implementation mechanism, and its alignment with broader environmental objectives, fitting well within the "Directive" category.

**Assigned Category:** Directive

## Instructions for Classification
When analyzing an input text, adhere to the following steps:
1. **Read and Comprehend the Text:** Ensure a thorough understanding of the policy details, regulatory standards, and implementation guidelines discussed.
2. **Evaluate Against Criteria:** Assess whether the text details a formal legislative directive or a regulation, considering the characteristics described in the respective category description.
3. **Make Your Classification:** Based on the content and context provided in the text, classify it into the appropriate category.

Please classify the following input text into the appropriate category:

{input}
"""

prompt_LDD_description_classification = """To accurately classify the input text, first understand its content and context, then match it to the most appropriate category based on the descriptions provided below:

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

Please classify the following input text into the appropriate category based on the descriptions above:

{input}
"""

prompt_LDD_COT_classification = """To ensure accurate classification of the input text into the appropriate category, follow these structured reasoning steps: 
1. **Analyze Input Text**: Begin by closely examining the input text. Identify key ideas and specific terminologies that relate closely to the categories listed below. Pay attention to unique concepts or phrases indicative of particular fields.
2. **Review Category Descriptions**: Read and understand the scope and focus of each category. Here are the descriptions to guide you:
   - 'cs.AI': Focuses on Artificial Intelligence, involving algorithms, machine learning techniques, and AI applications.
   - 'cs.CE': Encompasses Computational Engineering, including simulations, system design, and computational methods in engineering.
   - 'cs.CV': Pertains to Computer Vision, dealing with image recognition, processing, and computer-based visual tasks.
   - 'cs.DS': Involves Data Structures, emphasizing data organization, management, and storage structures.
   - 'cs.IT': Related to Information Theory, covering data transmission, encoding, and compression methodologies.
   - 'cs.NE': Deals with Neural and Evolutionary Computing, focusing on neural networks, genetic algorithms, and adaptive systems.
   - 'cs.PL': Concerns Programming Languages, including syntax, semantics, and language design.
   - 'cs.SY': Relates to Systems and Control, focusing on control systems, automation, and system dynamics.
   - 'math.AC': Involves Commutative Algebra, dealing with rings, modules, and algebraic structures.
   - 'math.GR': Pertains to Group Theory, focusing on mathematical groups, symmetries, and group structures.
   - 'math.ST': Deals with Statistics Theory, focusing on statistical methods and applications.
3. **Match Themes with Categories**: Compare the identified themes and terminologies from the input text with the provided category descriptions. Evaluate which category aligns best with the content of the text.
4. **Classify and Justify**: Conclude by categorizing the text into the most fitting category based on your analysis. Provide a detailed justification for your choice, explaining how the themes and terminologies from the text align with the category's focus.

Input text: {input}
"""


prompt_LDD_few_shots_classification = """
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

