"""
Testing the OpenAI LLM class.

Alexandre Pasquiou - November 2023
"""

import json
from model_inference.text_generation import OpenAILLM

function_1 = {
    "name": "bark",
    "description": "Whether to bark when you see a cat.",
    "args": [
        {
            "name": "target",
            "type": "string",
            "description": "the target cat",
            "required": "yes",
        },
        {
            "name": "blind",
            "type": "bool",
            "description": "whether you are blind and cannot see the cat.",
            "required": "yes",
        },
    ],
}


def test_init():
    """
    Testing the initialization of the OpenAILLM class
    """
    model = "gpt-3.5-turbo"
    llm = OpenAILLM(model)
    assert llm.model == model
    assert llm.self_hosted == False

    try:
        model = "gpt-AGI-open-source"
        llm = OpenAILLM(model)
    except Exception as e:
        assert e


def test_format_prompt():
    """
    Simply testing here the fact that the formatting functions
    correctly puts the tags in the prompt that will be sent to the OpenAILLM.
    """
    model = "gpt-3.5-turbo"
    llm = OpenAILLM(model)
    system_prompt = "You are a dog, woof woof !"
    user_prompt = "Meow ?"
    function_str = json.dumps(function_1)
    role = "assistant"
    function_prompt = "You have access to the following functions: "
    function_prompt += function_str
    function_prompt += (
        ". To call a function, respond - immediately and only - with a JSON object of the following format: "
        + llm.function_calling_format
        + ". Only use the functions if required !"
    )

    formatted_prompt = llm.format_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        functions=function_1,
        role=role,
    )
    assert formatted_prompt[0] == {
        "role": "system",
        "content": " ".join([system_prompt, function_prompt]),
    }
    assert formatted_prompt[1] == {"role": role, "content": user_prompt}


def test_generate():
    """
    Test text generation with openai models.
    """
    model = "gpt-3.5-turbo"
    llm = OpenAILLM(model)
    system_prompt = "You are a dog which is not blind. When you see a cat you bark. Otherwise you just respond: 'STARE', without calling a function."
    # Scenario 1: you see cat
    role = "assistant"
    history = None
    user_prompt = "Meow?"
    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        functions=function_1,
        role=role,
        history=history,
    )

    expected_answer = json.dumps(
        {"function": "bark", "arguments": {"target": "cat", "blind": False}}
    )
    assert response == expected_answer

    # Scenario 2
    role = "assistant"
    history = None

    try:
        user_prompt = "You see a cat."
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            functions=function_1,
            role=role,
            history=history,
        )
        assert response == "STARE"
    except AssertionError as e:
        assert response == expected_answer

    user_prompt = "You see a mouse."
    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        functions=function_1,
        role=role,
        history=history,
    )
    expected_answer = "STARE"
    assert response == expected_answer


# def test_generate_with_history():
#    """
#    Testing the generation with a conversation history.
#    """
#    model = "gpt-3.5-turbo"
#    llm = OpenAILLM(model)
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
