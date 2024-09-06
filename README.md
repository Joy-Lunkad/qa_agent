# Simple QA agent with limited tool use

The agent can extract answers from the context of a large pdf document, and optionally post it on a slack channel, using OpenAI LLMs.

### Ways to improve accuracy

1. Instead of using the question text to retrieve documents, ask the LLM to generate the retrieval query text.
2. Better RAG performance
   - Check for hallucinations, either with basic string matching or use an LLM. Use pydantic structured output for citations.
   - Tune the chunk size parameter.
   - Increase top K retrieval.
3. Better prompts - clearer instructions
4. Use bigger, powerful LLMs leveraging frameworks like llama-index

### Increasing Modularity and scalability

1. Use Faster Datastores, eg, Pinecone
2. Store data in cloud buckets instead of locally
3. Modularity
   - PromptTemplate classes
4. Used too many list comprehensions, should replace with parallel_map.
5. Proper Typing, using Enums instead of strings.
6. Docker instead of a requirements.txt