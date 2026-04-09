## Task Description

You are an average news reader. you will be provided a full news article.  
From a reader’s perspective, describe your immediate impression of the news and make reasonable inferences at the detail level.

### Task Requirements
You need to complete the following two parts:


**Article Interpretation**  
   - Analyze **only** based on the news article.  
   - Describe what you see (surface interpretation).  
   - Infer what event might be happening based on the news article (event implication).  
   - do **not** use your own world knowledge.  


### Input

Full News Context: {{CONTEXT}}




### Output Format (JSON)
Response With english
{
    "News_Context": {
        "Surface_Interpretation": "What is the surface interpretation?",
        "Event_Implication": "What is the deep meaning, and what is the purpose?"
    }
}
