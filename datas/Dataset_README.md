## Dataset Format

Each instance in the dataset is stored as a JSON object.

### Fields for Misleading Detection

The misleading detection data is provided in `train_data.json` and `test_data.json`.

| Field | Description |
|---|---|
| `caption` | The textual modality in the news preview. |
| `article_context` | The contextual information from the full news article. |
| `image_path` | The path to the image used in the news preview. |
| `Reader_View_ih` | **Up**: The LLM-simulated reader’s understanding of the news preview. |
| `Reader_View_context` | **Uc**: The LLM-simulated reader’s understanding of the article context. |
| `Misleading` | The misleading detection result, including the predicted label and an explanatory rationale. |

### Fields for Misleading Correction

The misleading correction data is provided in `test_correction.json`.

| Field | Description |
|---|---|
| `caption` | The textual modality in the news preview. |
| `article_context` | The contextual information from the full news article. |
| `image_path` | The path to the image used in the news preview. |
| `Reader_View_ih` | **Up**: The LLM-simulated reader’s understanding of the news preview. |
| `Reader_View_context` | **Uc**: The LLM-simulated reader’s understanding of the article context. |
| `Misleading` | The misleading detection result, including the predicted label and an explanatory rationale. |
| `free_form` | A free-form rewrite of the original caption. |
| `mini-edit` | A constrained rewrite of the original caption, with restrictions on style, length, or editing scope. |

The `free_form` and `mini-edit` fields contain the following subfields:

| Field | Description |
|---|---|
| `Misleading_Cause` | The misleading cues identified based on the rationale from misleading detection. |
| `Suggested_Improvement` | Suggestions for correcting or improving the headline/caption. |
| `Rewriten_Headline` | The corrected version of the original caption. |
| `Rewriten_Reader_View` | The LLM-simulated reader’s understanding of the corrected news preview. |
| `Rewriten_Misleading` | The misleading detection result for the corrected news preview. |