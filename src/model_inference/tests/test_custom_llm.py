"""
Testing the CustomLLM class.

Alexandre Pasquiou - December 2023
"""

import json
from dataset_generation.text_generation import CustomLLM


function_1 = {
    "name": "search_product",
    "description": "When you want to find a product using a 'use_case', a 'price_range', a 'category', a 'product' or 'specifications'.",
    "args": [
        {
            "name": "use_case",
            "type": "string",
            "description": "how to use the product.",
            "required": "no",
        },
        {
            "name": "price_range",
            "type": "str",
            "description": "the price to pay for the product",
            "required": "no",
        },
        {
            "name": "category",
            "type": "str",
            "description": "the category of the product",
            "required": "no",
        },
        {
            "name": "product",
            "type": "str",
            "description": "the name of the product",
            "required": "no",
        },
        {
            "name": "specifications",
            "type": "str",
            "description": "product specifications, that is its characteristics",
            "required": "no",
        },
    ],
}


def test_init():
    """
    Testing the initialization of the CustomLLM class
    """
    model = "sentence-transformers/all-MiniLM-L6-v2"
    endpoint_ip = "1.1.1.1"
    port = 8080
    llm = CustomLLM(model, self_hosted=True, endpoint_ip=endpoint_ip, port=port)
    assert llm.model == model
    assert llm.self_hosted == True
    assert llm.endpoint_ip == endpoint_ip
    assert llm.port == port

    try:
        model = "gpt-AGI-open-source"
        llm = CustomLLM(model)
    except Exception as e:
        assert e


def test_format_prompt():
    """
    Simply testing here the fact that the formatting functions
    correctly puts the tags in the prompt that will be sent to the CustomLLM.
    """
    model = "sentence-transformers/all-MiniLM-L6-v2"
    endpoint_ip = "1.1.1.1"
    port = 8080
    llm = CustomLLM(model, self_hosted=True, endpoint_ip=endpoint_ip, port=port)
    # Prompting
    system_prompt = (
        "You are an helpful AI assistant that helps human on e-commerce websites."
        + "You answer their inquiries relatives to smartphones and make relevant product recommendations when needed."
    )
    user_prompt = "I am looking for a new phone."
    function_str = json.dumps(function_1)
    B_FUNC, E_FUNC = " <FUNCTIONS>", "</FUNCTIONS>\n"
    B_INST, E_INST = "[INST]", "[/INST]\n\n"
    B_SYS, E_SYS = "<SYS>", "</SYS>\n\n"
    function_prompt = "You have access to the following functions: " + B_FUNC
    function_prompt += function_str
    function_prompt += (
        E_FUNC
        + " To call a function, respond - immediately and only - with a JSON object of the following format: "
        + llm.function_calling_format
        + ". Only use the functions if required !"
    )

    formatted_prompt = llm.format_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        functions=function_1,
    )
    expected_response = (
        B_INST
        + B_SYS
        + system_prompt
        + " "
        + function_prompt
        + E_SYS
        + user_prompt.strip()
        + E_INST
    )
    assert formatted_prompt == expected_response


def test_generate():
    """
    Test text generation with openai models.
    """
    model = "optimum/roberta-base-squad2"
    endpoint_ip = "1.1.1.1"
    port = 8080
    llm = CustomLLM(model, self_hosted=True, endpoint_ip=endpoint_ip, port=port)

    system_prompt = (
        "You are an helpful AI assistant that helps human on e-commerce websites."
        + "You answer their inquiries relatives to smartphones and make relevant product recommendations when needed."
    )  # Scenario 1: you see cat
    history = None
    user_prompt = "I am looking for a new iphone."

    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        functions=function_1,
        history=history,
    )

    expected_answer = json.dumps(
        {
            "function": "search_product",
            "arguments": {
                # "use_case": None,
                # "price_range": None,
                "category": "smartphone",
                "product": "iphone",
                # "specifications": None
            },
        }
    )
    assert response == expected_answer


# def test_generate_with_history():
#    """
#    Testing the generation with a conversation history.
#    """
#    model = "gpt-3.5-turbo"
#    llm = CustomLLM(model)
#    system_prompt = "You are a dog which is not blind. When you see a cat you bark. Otherwise you just respond: 'STARE', without calling a function."
#    # Scenario 1: you see cat
#    role = "assistant"
#    history = [
#        {"role": "assistant", "content": "STARE"},
#        {"role": "user", "content": "A cat is passing by."},
#    ]
#    user_prompt = "Meow?"
#    response = llm.generate(
#        system_prompt=system_prompt,
#        user_prompt=user_prompt,
#        functions=function_1,
#        role=role,
#        history=history,
#    )
#    expected_answer = json.dumps(
#        {"function": "bark", "arguments": {"target": "cat", "blind": False}}
#    )
#    assert response == expected_answer
#
#    # Scenario 2
#    role = "assistant"
#    history = [
#        {"role": "assistant", "content": "STARE"},
#        {"role": "user", "content": "A mouse is passing by."},
#    ]
#
#    try:
#        user_prompt = "You see a cat."
#        response = llm.generate(
#            system_prompt=system_prompt,
#            user_prompt=user_prompt,
#            functions=function_1,
#            role=role,
#            history=history,
#        )
#        assert response == "STARE"
#    except AssertionError as e:
#        assert response == expected_answer
#
#    user_prompt = "You see a mouse."
#    response = llm.generate(
#        system_prompt=system_prompt,
#        user_prompt=user_prompt,
#        functions=function_1,
#        role=role,
#        history=history,
#    )
#    expected_answer = "STARE"
#    assert response == expected_answer
