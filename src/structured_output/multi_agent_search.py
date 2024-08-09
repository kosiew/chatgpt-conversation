import json
from enum import Enum
from io import StringIO
from re import A

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from duckduckgo_search import DDGS
from icecream import ic
from IPython.display import Image

# openai's BaseModel require Config.extra = "forbid" to ensure no additional properties are allowed
# we're also using Config to add descriptions to the pydantic models
from openai import BaseModel, OpenAI, pydantic_function_tool
from pydantic import Field

client = OpenAI()
MODEL = "gpt-4o-2024-08-06"
duck = DDGS()
