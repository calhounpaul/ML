import os, sys, json, re, time, datetime, random, string, requests, hashlib, multiprocessing, traceback

parent_dir = os.path.dirname(os.path.realpath(__file__))
docker_data_dir = parent_dir


def one_curl_request(prompt, max_tokens=4096,url="http://127.0.0.1:8000/generate",opts={}):
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    data.update(opts)
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

MISTRAL_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "codify_recurrence",
            "description": "Define the recurrence based on precise parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "freq": {
                        "type": "string",
                        "enum": [
                            "each_year",
                            "each_month",
                            "each_week",
                            "each_day",
                            "each_hour",
                            "each_minute",
                            "each_second"
                        ],
                        "description": "The type of frequency for the recurrence. E.g. 'each_year' is for a yearly recurrence."
                    },
                    "start_year": {
                        "type": "integer",
                        "minimum": 2024,
                        "maximum": 2100,
                        "description": "The year that the recurrence starts."
                    },
                    "start_month": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12,
                        "description": "The month that the recurrence starts."
                    },
                    "start_day": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 31,
                        "description": "The day that the recurrence starts."
                    },
                    "start_hour": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 23,
                        "description": "The hour that the recurrence starts."
                    },
                    "start_minute": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 59,
                        "description": "The minute that the recurrence starts."
                    },
                    "interval_between_recurrences": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                        "maximum": 100,
                        "description": "The interval between recurrences. E.g. 2 for 'every other year'."
                    },
                    "day_that_week_starts_on": {
                        "type": "string",
                        "enum": [
                            "sunday",
                            "monday",
                            "tuesday",
                            "wednesday",
                            "thursday",
                            "friday",
                            "saturday"
                        ],
                        "description": "The day the week starts on"
                    },
                    "continue_for_this_many_cycles": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "The number of occurrences after which the recurrence cycles end."
                    },
                    "end_year": {
                        "type": "integer",
                        "minimum": 2024,
                        "maximum": 2100,
                        "description": "The year that the recurrence ends."
                    },
                    "end_month": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12,
                        "description": "The month that the recurrence ends."
                    },
                    "end_day": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 31,
                        "description": "The day that the recurrence ends."
                    },
                    "end_hour": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 23,
                        "description": "The hour that the recurrence ends."
                    },
                    "by_the_nth_occurrence_position": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": -366,
                            "maximum": 366,
                            "not": 0
                        },
                        "description": "Specific occurrences to include in the recurrence set. Cannot be 0. More than one value can be provided. E.g. [1, -1] for the first and last day of the month. [30,60,90] for the 30th, 60th, and 90th day of the year. Always used in conjunction with one of the other 'by' parameters."
                    
                    },
                    "by_month": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 12
                        },
                        "description": "Months to include in the recurrence set. More than one value can be provided. E.g. [1, 3, 5] for January, March, and May."
                    },
                    "by_month_day": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": -31,
                            "maximum": 31,
                            "not": 0
                        },
                        "description": "Days of the month to include in the recurrence set. Negative values count from the end of the month. More than one value can be provided. E.g. [1, 15, -1] for the 1st, 15th, and last day of the month."
                    },
                    "by_year_day": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": -366,
                            "maximum": 366,
                            "not": 0
                        },
                        "description": "Days of the year to include in the recurrence set. Negative values count from the end of the year. More than one value can be provided. E.g. [1, 100, -1] for the 1st, 100th, and last day of the year."
                    },
                    "by_easter": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": -366,
                            "maximum": 366,
                        },
                        "description": "Days relative to Easter to include in the recurrence set. More than one value can be provided. E.g. [-1, 0, 1] for the day before Easter, Easter, and the day after Easter."
                    },
                    "by_week_number": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": -53,
                            "maximum": 53,
                            "not": 0
                        },
                        "description": "Weeks of the year to include in the recurrence set. Negative values count from the end of the year. More than one value can be provided. E.g. [1, 26, -1] for the 1st, 26th, and last week of the year."
                    },
                    "by_weekday": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "sunday",
                                "monday",
                                "tuesday",
                                "wednesday",
                                "thursday",
                                "friday",
                                "saturday"
                            ]
                        },
                        "description": "Days of the week to include in the recurrence set. More than one value can be provided. E.g. ['monday', 'wednesday', 'friday'] for Monday, Wednesday, and Friday of the week."
                    },
                    "by_hour": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 23
                        },
                        "description": "Hours of the day to include in the recurrence set. More than one value can be provided. E.g. [0, 12, 18] for midnight, noon, and 6 PM."
                    },
                    "by_minute": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 59
                        },
                        "description": "Minutes of the hour to include in the recurrence set. More than one value can be provided. E.g. [0, 15, 30, 45] for the start of the hour and every quarter hour."
                    },
                    "by_second": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 59
                        },
                        "description": "Seconds of the minute to include in the recurrence set. More than one value can be provided. E.g. [0, 15, 30, 45] for the start of the minute and every quarter minute."
                    }
                },
                "required": [
                    "freq",
                    "start_year",
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "format": {
                        "type": "string",
                        "enum": [
                        "celsius",
                        "fahrenheit"
                        ],
                        "description": "The temperature unit to use. Infer this from the users location."
                    }
                },
                "required": [
                "location",
                "format"
                ]
            }
        }
    }
]

from dateutil.rrule import rrule, rrulestr
from datetime import datetime

def generate_rrule(arguments):
    freq_map = {
        "each_year": "YEARLY",
        "each_month": "MONTHLY",
        "each_week": "WEEKLY",
        "each_day": "DAILY",
        "each_hour": "HOURLY",
        "each_minute": "MINUTELY",
        "each_second": "SECONDLY"
    }
    
    weekday_map = {
        "monday": "MO",
        "tuesday": "TU",
        "wednesday": "WE",
        "thursday": "TH",
        "friday": "FR",
        "saturday": "SA",
        "sunday": "SU"
    }
    
    freq = freq_map[arguments["freq"]]
    dtstart = datetime(arguments["start_year"], arguments["start_month"], arguments["start_day"], arguments.get("start_hour", 0), arguments.get("start_minute", 0))
    until = datetime(arguments["end_year"], arguments["end_month"], arguments["end_day"])
    
    byweekday = [weekday_map[day] for day in arguments.get("by_weekday", [])]
    interval = arguments.get("interval_between_recurrences", 1)
    count = arguments.get("continue_for_this_many_cycles", None)
    
    rrule_string = f"FREQ={freq};DTSTART={dtstart.strftime('%Y%m%dT%H%M%S')};INTERVAL={interval};"
    
    if byweekday:
        rrule_string += f"BYDAY={','.join(byweekday)};"
    
    if count:
        rrule_string += f"COUNT={count};"
    
    rrule_string += f"UNTIL={until.strftime('%Y%m%dT%H%M%S')}"
    
    return rrule_string

class MistralFunct:
    def __init__(self, funct_name):
        for funct in MISTRAL_FUNCTIONS:
            if funct["function"]["name"] == funct_name:
                self.name = funct_name
                self.json = funct
                break
        assert self.name is not None, f"Function {funct_name} not found in MISTRAL_FUNCTIONS"
    
    def get_prompt(self, instruction):
        return f"[AVAILABLE_TOOLS] {json.dumps(MISTRAL_FUNCTIONS)}[/AVAILABLE_TOOLS][INST] {instruction} [/INST][TOOL_CALLS]"


def runtest(instruction):
    prompt = MistralFunct("codify_recurrence").get_prompt(instruction)
    #print(prompt)
    #input()
    responses = one_curl_request(prompt)
    #print(responses)
    returned_json = json.loads(responses["text"][0][len(prompt):])
    print(json.dumps(returned_json, indent=4))
    for tool_use in returned_json:
        tool_args = tool_use["arguments"]
        rrule_out = generate_rrule(tool_args)
        print(rrule_out)


if __name__ == "__main__":
    list_of_instructions = [
        "Create a recurrence rule that starts on January 2, 2024, at 11:00 AM, repeats weekly on Tuesdays and Fridays, and ends on December 30, 2024.",
        "Create a recurrence rule that runs every first Monday of the month starting on February 1, 2024 and ending on November 30, 2024.",
        "Create a recurrence rule that runs every alternate Saturday starting on March 1, 2024 and ending on December 28, 2024.",
        "Provide a recurrence rule that starts on January 3, 2024, at 12:00 PM, repeats weekly on Wednesdays and Saturdays, and ends on December 29, 2024.",
        "Create a recurrence rule that starts on January 4, 2024, at 1:00 PM, repeats weekly on Thursdays and Sundays, and ends on December 26, 2024.",
        "Create a recurrence rule that runs every second Tuesday of the month starting on April 2, 2024 and ending on December 10, 2024.",
        "Create a recurrence rule that runs every other Monday starting on May 6, 2024 and ending on December 30, 2024.",
        "Provide a recurrence rule that starts on January 5, 2024, at 2:00 PM, repeats weekly on Fridays and Mondays, and ends on December 27, 2024.",
        "Create a recurrence rule that starts on January 6, 2024, at 3:00 PM, repeats weekly on Saturdays and Tuesdays, and ends on December 28, 2024.",
        "Create a recurrence rule that runs every third Wednesday of the month starting on June 19, 2024 and ending on December 18, 2024.",
        "Create a recurrence rule that runs every alternate Tuesday starting on July 2, 2024 and ending on December 24, 2024.",
        "Provide a recurrence rule that starts on January 7, 2024, at 4:00 PM, repeats weekly on Sundays and Wednesdays, and ends on December 29, 2024."
    ]

    for instruction in list_of_instructions:
        print(instruction)
        runtest(instruction)