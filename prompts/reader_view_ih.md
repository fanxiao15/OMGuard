## Task Description

You are an average news reader. you will be provided a piece of news that includes an image, a news headline.  
From a reader’s perspective, describe your immediate impression of the news and make reasonable inferences at the detail level.

### Task Requirements
You need to complete the following two parts:

**Image–Headline Interpretation**  
   - Analyze **only** based on the image and the news headline.  
   - Describe what you see (surface interpretation).  
   - Infer what event might be happening based on visual cues and the headline (event implication).  
   - Do **not** refer to the full news context, and do **not** use your own world knowledge.  

### Input
News Headline: {{NEWS_HEADLINE}}

Image: (will be provided)

### Output Format (JSON)
Response With english
{
    "Image–Headline": {
        "Surface_Interpretation": "What is the surface interpretation?",
        "Event_Implication": "What is the deep meaning, and what is the purpose?"
    }
}
