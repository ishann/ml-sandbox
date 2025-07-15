# text2ppt

CLI tool to convert text into a structured PowerPoint presentation.
Usage: 
```python main.py --data ./data/inputs/Transformers.pdf --meta ./data/inputs/mission_statement.txt```

**Local image generation using SDXL has been reduced in fidelity to
allow for reasonable run-times when demo-ing.** See `./generator/image_generator.py/generate_image_local` usage to increase fidelity.

NEEDS:
1. `ollama` running locally to summarize text and return slide content.
   1. `Mixtral-8b` requires 30-35Gb memory.
2. `SDXL` running through `diffusers`
3. (Optional) OpenAI API key on paid plan to generate images using dall-e-3.

TODOs:
1. Document ingestions needs significant improvements, especially to handle images/ tables/ equations.
2. The text summarization prompt needs the following
   1. Input guardrails to make sure non-sensical text extraction doesn't get to the LLM.
   2. Output guardrails so that the tool does not generate non-sensical output text.
3. The layout of the entire presentation should be modeled as a graph where within-slide content is a clique and across-slide content is sparsely connected. Node attributes have {x,y,h,w,font,color,etc.}. See `./generator/layout_engine.py` for a sketch of what this will look like.
4. Refactor code. Bottom-up vibe-coding has resulted in OOP principles not being followed.

TAKEAWAYS:
1. Locally running SDXL models can be suprisingly good for narrow tasks. 
2. python-pptx is surprisingly poor with simple tasks.
3. Prompt engineering is an art and also a craft. It requires persistence.
4. Using Mixtral-8b to engineer prompts for SDXL results in significant improvements.