from langchain_core.prompts import ChatPromptTemplate

# Shared system instructions used across all model types
_SYSTEM_INSTRUCTIONS = """You are an expert assistant for the Stanford Research Computing Center (SRCC).
Always prioritize the provided documentation when answering — it reflects how these clusters are actually configured.
- HPC clusters are highly customized. Partitions, resource limits, job scheduling policies, storage paths, software modules, and local tools often differ from defaults. When the documentation addresses a topic, use it as the authoritative source and cite it using [Title] where Title is the exact title from the document's metadata.
- General knowledge may supplement your answers when the documentation is silent — for instance, explaining a concept the docs reference but don't define. Blend naturally rather than rigidly separating documented and general knowledge.
- If a question is about cluster-specific details not covered in the documentation, direct the user to srcc-support@stanford.edu rather than guessing.
- Prioritize clear, actionable steps. Use bulleted lists for multi-step procedures.
- Answer the user's query directly. Do not add extra information or conversational text after the answer is complete."""


def get_prompt_template(model_type: str) -> ChatPromptTemplate:
    """
    Return appropriate prompt template based on model type.

    Args:
        model_type: Type of model ('llama', 'gemma', etc.)

    Returns:
        ChatPromptTemplate configured for the model type
    """

    if model_type == "llama":
        # Llama uses [INST] tags; rag_service.py also wraps via apply_chat_template
        return ChatPromptTemplate.from_template(
            "<s>[INST] " + _SYSTEM_INSTRUCTIONS + "\nCONTEXT:\n{context}\n\nUSER QUERY:\n{query} [/INST]"
        )
    else:
        # Qwen, Gemma, and other models: plain template.
        # For Qwen/Llama-family, rag_service._generate_response wraps this via
        # tokenizer.apply_chat_template, so the system instructions here become
        # the body of the user turn and are still followed correctly.
        return ChatPromptTemplate.from_template(
            _SYSTEM_INSTRUCTIONS + "\nCONTEXT:\n{context}\n\nUSER QUERY:\n{query}"
        )
