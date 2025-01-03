from .common import *
from .preprocessing import get_colored_mask, DataGenerator
from .model.load_model import load_model


def segment(IMG_PATH,path=[]):

    images_data = DataGenerator(
        IMG_PATH,
        img_files=path,        
        batch_size=1,          
        size=(512, 512),
        workers=4, 
        use_multiprocessing=True
        )

    imgs  = images_data.__getitem__(0)
    model = load_model()


    img = np.expand_dims(imgs[0], axis=0)
    pred = (np.squeeze(model.predict(img,verbose=0))*255).astype(np.uint8)
    img = (np.squeeze(img) * 255).astype(np.uint8)

    image = cv2.imread(os.path.join(IMG_PATH, path[0]))

    height, width, _ = image.shape

    resized_mask = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)


    return get_colored_mask(image,resized_mask,color = [255,157,61])





if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_PATH = os.path.join(BASE_DIR, "imgs/",)  

    img_files = sorted(os.listdir(IMG_PATH))
    try:
        result = segment(IMG_PATH,path=img_files)
        print("Segmentation result:", result.shape)
        cv2.imshow('Image Window', result)

        # Wait for a key press indefinitely or for a specified amount of time in milliseconds
        cv2.waitKey(0)

        # Close all OpenCV windows
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")

