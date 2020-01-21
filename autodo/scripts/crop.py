import pandas as pd
import pyvips
from os import makedirs
from os.path import join
from prettyparse import Usage

from autodo.dataset import Dataset
from autodo.scripts.base_script import BaseScript


class CropScript(BaseScript):
    usage = Usage('''
        Crop images based on network output csv

        :data_csv str
            Network output csv
        
        :dataset_folder str
            Dataset folder

        :out_folder str
            Output folder to write cropped images to
        
        :-t --test
            Use test images
        
        :-s --size str 3380,2710
            Bounding box image size
    ''', size=lambda x: tuple(map(int, x.size.split(','))))

    def run(self):
        args = self.args
        dataset = Dataset.from_folder(args.dataset_folder)
        makedirs(args.out_folder, exist_ok=True)
        source_folder = dataset.images_folder[1 if args.test else 0]
        df = pd.read_csv(args.data_csv)
        for image_id, rows in df.groupby('image_id'):
            image_file_name = join(source_folder, image_id + '.jpg')
            for i, (_, row) in enumerate(rows.iterrows()):
                out_file_name = join(args.out_folder, image_id + '-{:02}.jpg'.format(i))
                img = pyvips.Image.new_from_file(image_file_name, access='sequential')  # type: pyvips.Image
                width = img.width - 1
                height = img.height - 1
                w, h = args.size
                xmin = max(0.0, min(1.0, row['xmin'] / w))
                ymin = max(0.0, min(1.0, row['ymin'] / h))
                xmax = max(0.0, min(1.0, row['xmax'] / w))
                ymax = max(0.0, min(1.0, row['ymax'] / h))
                cropped = img.crop(
                    int(width * xmin),
                    int(height * ymin),
                    max(1, int(width * (xmax - xmin))),
                    max(1, int(height * (ymax - ymin))),
                )
                cropped.write_to_file(out_file_name)
                print('Wrote to {}.'.format(out_file_name))


main = CropScript.run_main
