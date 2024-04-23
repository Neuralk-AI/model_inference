"""
Testing the BaseLLM class.

Alexandre Pasquiou - November 2023
"""

import json
from dataset_generation.text_generation import BaseLLM, call_function

function_dict = {
    "function": "function_name",
    "arguments": {
        "argument1": "value1",
        "argument2": "value2",
        "argument3": "value3",
    },
}
function_calling_format = json.dumps(function_dict)

function_input_format = {
    "name": "find_apples",
    "description": "it is used to find apples",
    "args": [
        {
            "name": "apple_name",
            "type": "string",
            "description": "the name of the apple",
            "required": "yes",
        },
        {
            "name": "apple_color",
            "type": "string",
            "description": "the color of the apple",
            "required": "no",
        },
    ],
}


def test_init():
    """
    Testing the initialization of the class BaseLLM.
    """
    model = "gpt-3.5-turbo"
    llm = BaseLLM(model)
    assert llm.model == model
    assert llm.function_calling_format == function_calling_format
    assert llm.function_input_format == function_input_format


def test_call_function():
    """
    Testing the `call_function` method.
    """
    x, y, z, label = 5, "3", 10, "salamander"
    call = {
        "function": "function2call",
        "arguments": {
            "x": x,
            "y": y,
            "z": z,
            "label": label,
        },
    }
    function_call = json.dumps(call)
    output = call_function(function_call)
    assert output[0] == 5 + 3 + 10
    assert isinstance(output[0], float)
    assert output[1] == 5 * 3 * 10
    assert isinstance(output[1], float)
    assert output[2] == label + label
    assert isinstance(output[2], str)
