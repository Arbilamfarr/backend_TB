import os

def load_weights(model,model_file='model.h5'):
    pretrained_modul = os.path.join(os.path.dirname(os.path.abspath(__file__)) , model_file) 
    print(pretrained_modul)
    model.load_weights(pretrained_modul)
    return model