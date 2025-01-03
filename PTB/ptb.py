import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import numpy as np
from .img_pretraining import DataGenerator
from .img_pretraining import convert_to_rgb
from .model import create_model
from .model import load_weights

def predict_PTB(IMAGE_PATH,path=[]):



    images = DataGenerator(
        IMAGE_PATH,
        img_files=path,        
        batch_size=8,          
        size=(96, 96),
        workers=4, 
        use_multiprocessing=True)
    
    imgs,targets  = images.__getitem__(0)


    imgs = convert_to_rgb(imgs)

    model = create_model()
    
    model = load_weights(model)

    img = np.expand_dims(imgs[0], axis=0)
    
    # preds = [val_loss, val_acc]
    [preds] = model.predict(img,verbose=0)

    if __name__ == '__main__':
        print("[val_loss, val_acc] ",preds)
        print("Target ",targets)
    
    return preds



if __name__ == '__main__':
    # Example usage
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    IMAGE_PATH = os.path.join(BASE_DIR, "imgs/") 
    img_files = sorted(os.listdir(IMAGE_PATH))
    predictions = predict_PTB(IMAGE_PATH,path=img_files)