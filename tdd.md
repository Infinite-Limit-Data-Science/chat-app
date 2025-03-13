# Technical Overview

Part of the lack of capability in the open-source, leading to challenges of implementation of core features of GuideWell Chat, our development team, leveraging baseline NLP packages, such as langchain, nltk, numpy, scipy, sklearn, pandas, matplotlib, and seaborn, developed the capability ourselves. The structure and organization of the code, the programming interfaces, and the abstractions are designed around design patterns and techniques inherit in the langchain_core python source code, which we read in its entirety. In effect, we developed 6 generic packages intended for reuse for NLP tasks, RAG pipelines, unstructured and semistructured document ingestion pipelines, state management, single and multi-agent workflows, history management, and more. 

## GWBlue HuggingFace (gwblue_huggingface)

The gwblue_huggingface package is broken down to six core modules:
- HuggingFace Chat Model
- HuggingFace Embeddings
- HuggingFace Inference Client (with support for HuggingFace TGI, TEI, and vLLM)
- HuggingFace LLM
- HuggingFace Transformer Tokenizers

Below is a technical break down of each:

### HuggingFace Chat Model




and general LLM abstractions.

