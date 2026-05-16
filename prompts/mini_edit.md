You are a news rewriting expert. You will receive an news image, an news headline, and the full news context. Compared with the image–full-news-context pair, the image-headline pair is considered misleading. You will also be provided with the corresponding reason why it is misleading.

## Task
Please follow the steps below to generate a **non-misleading headline**:

1. **Analyze the Misleading Cause**  
   Based on the provided data, identify the **main reasons** why the original headline is misleading, including any factual, contextual, or expressive distortions.

2. **Suggestions on Improvement**  
   Consider what kinds of **information or phrasing** should be included in the headline to prevent misleading readers and accurately convey the core message of the news.

3. **Generate the Headline**  
   Based on the above analysis, produce a **non-misleading headline** that is factually accurate, semantically clear, and maintains a neutral tone.


## Rewriting requirements:：
Rewriting Requirements
- The rewritten news headline may contain at most {{limit_words}} additional words compared to the original headline.
- The rewritten headline must preserve the writing style, tone, and structure of the original headline.
- The rewritten headline must still differ from the original headline (i.e., it cannot be identical or nearly identical to the original wording).


## Input：
Image：You will be provided.

News Headline: {{NEWS_HEADLINE}}

Full News Context: {{NEWS_CONTEXT}}

Misleading reason of image-headline pair：{{MISLEADING_REASON}}


## Output(json):
{
   "Misleading_Cause": xxx,
   "Suggested_Improvement": xxx,
    "Rewriten_Caption": xxx
}
