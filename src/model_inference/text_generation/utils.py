"""
Alexandre Pasquiou - November 2023
"""

import json


def call_function(json_str):
    """Execute function defined in JSON format."""
    # Parse JSON string
    function_dict = json.loads(json_str)
    # Get function name
    function_name = function_dict["function"]
    # Get arguments
    arg_dict = function_dict["arguments"]

    # Call function dynamically
    function = globals().get(function_name)
    if function:
        return function(**arg_dict)
    else:
        raise NotImplementedError(f"Function '{function_name}' not found")
