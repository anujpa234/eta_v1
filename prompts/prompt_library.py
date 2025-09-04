# Prepare prompt template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt for document analysis
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Prompt for document comparison
document_comparison_prompt = ChatPromptTemplate.from_template("""
You will be provided with content from two PDFs. Your tasks are as follows:

1. Compare the content in two PDFs
2. Identify the difference in PDF and note down the page number 
3. The output you provide must be page wise comparison content 
4. If any page do not have any change, mention as 'NO CHANGE' 

Input documents:

{combined_docs}

Your response should follow this format:

{format_instruction}
""")

# Prompt for contextual question rewriting
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
        "You are a document assistant that ONLY answers questions using the provided context. "
        "STRICT RULES:\n"
        "1. If the context contains information to answer the question: provide a clear answer\n"
        "2. If the context does NOT contain relevant information: respond EXACTLY with 'I don't have enough relevant information to answer your question.'\n"
        "3. DO NOT use your general knowledge\n"
        "4. DO NOT provide information not in the context\n"
        "5. DO NOT say 'However...' and then provide general information\n"
        "6. DO NOT be helpful beyond what the context provides\n\n"
        "You MUST respond in {language}.\n\n"
        "Context:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

PROMPT_REGISTRY = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}

DEFAULT_LANGUAGE = {
    "English"
}