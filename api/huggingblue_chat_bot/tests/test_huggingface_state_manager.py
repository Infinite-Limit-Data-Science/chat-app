from ..huggingface_graph import graph

initial_state = {
    "conversation_id": "my-convo-123",
    "datasource": "multidoc_compare",  # we already know the source
    "vectorstore_docs": [],
    "messages": [
        {
          "role": "user",
          "content": "'What is Generative AI?'"
        }
    ],
}

def test_guardrails_node():
    for event in graph.stream(initial_state):
        for value in event.values():
            assert value['messages'].strip('\n') == 'safe'