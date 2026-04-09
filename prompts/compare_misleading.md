## Task Description
You will receive:
- An image  
- A news headline  
- The full news context  
- A reader’s surface interpretation and event implication **for the image–headline pair**  
- A reader’s surface interpretation and event implication **for the full new context**


### Your Task
Based on the given interpretations:
- If a reader forms an impression about the nature, status, cause and effect, the responsible party, or severity of a news event when only exposed to images and titles, and this impression is significantly corrected, restricted, or overturned after reading the full news, it is considered to be misleading.

- On the contrary, if the full news only elaborates, extends, or supplements the content implied by the title (for example, by providing more details, reactions, or outcomes), without altering the reader's understanding of the basic direction or core judgment of the event, it is considered that there is no misleading.



### Input
Image: (will be provided)

News Headline: {{NEWS_HEADLINE}}

Full News Context: {{CONTEXT}}

Reader Interpretations: {{READER_INFER}}


### Output Format (JSON)
Response With english
{
    "Misleading": "Yes/No",
    "Reason": "Not less than 100 words, focus on event level, not social impact"
}

