import numpy as np
import matplotlib.pyplot as plt

from .lfpmodels import CoulombModel


first_model = CoulombModel("First")

LFP = first_model.produce_local_field_potential()