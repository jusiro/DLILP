name = "dlilp",
version = "0.1.0",

from .modeling.model import VLMModel
from .modeling.constants import CATEGORIES
from .modeling.prompts import generate_prompt_ensemble, CATEGORIES_ALL, ASSEMBLE_PROMPTS