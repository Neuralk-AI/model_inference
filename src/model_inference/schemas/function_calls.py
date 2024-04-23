"""
Base class for LLMs.

Alexandre Pasquiou - November 2023
"""

import json


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
