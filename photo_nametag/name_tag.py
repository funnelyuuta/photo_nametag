from photo_nametag.face import predict
from photo_nametag.exif import exif
from pathlib import Path
from PIL import UnidentifiedImageError

def name_tag_main(per):
    image_dir = Path('data','image_tag')
    image_paths = image_dir.glob('**/*')
    model_sav = Path('data','model','model.sav')
    model_csv = Path('data','model','model.csv')
    face_dir = Path('data',"model","face")
    for image_path in image_paths:
        name = image_path.stem
        suffix_ = image_path.suffix
        try:
            name_list = predict(image_path, name, suffix_, per, model_sav, model_csv, face_dir)
        except UnidentifiedImageError:
            print("画像が壊れてるため、スキップします")
            continue
        except ValueError:
            print("顔を検出できませんでした、スキップします")
            continue
        exif(image_path, name_list)

