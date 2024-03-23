import numpy as np
import time, datetime
import os, sys
from os.path import join as pjoin
import pandas as pd
from myutils import info, create_readme

STATE = pd.Series({'S': 0, 'I': 1, 'R': 2})
