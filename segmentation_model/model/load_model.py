from ..common import *
from .basic import jaccard_index,dice_coefficient

def load_model(best_model='best_model.keras'):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    BEST_MODEL = os.path.join(BASE_DIR, best_model)  
    return  tf.keras.models.load_model(BEST_MODEL,custom_objects={'dice_coefficient': dice_coefficient,'jaccard_index':jaccard_index})
    