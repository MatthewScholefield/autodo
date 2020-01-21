from collections import namedtuple

import pandas
from os.path import join
from prettyparse import Usage

from autodo.scripts.base_script import BaseScript
from autodo.stage_three_predictor import StageThreePredictor

StageThreeRow = namedtuple('StageThreeRow', 'image_id box_id x y z')


class RunStageThreeScript(BaseScript):
    usage = Usage('''
        Train the stage three nethreerk
        
        :model_file str
            Model file to load from
        
        :stage_two_file str
            Csv file with output from stage two
        
        :image_folder str
            Folder with source images
        
        :-g --gpu
            Train with GPU
        
        :-o --output-file str -
            Output csv to write to
    ''')

    def run(self):
        args = self.args
        boxes = pandas.read_csv(args.stage_two_file)
        predictor = StageThreePredictor(args.model_file, args.gpu)
        rows = []
        try:
            for image_id, image_rows in boxes.groupby('image_id'):
                image_filename = join(args.image_folder, '{}.jpg'.format(image_id))
                xcenters, ycenters = image_rows[['center_x', 'center_y']].values.T
                label = predictor.predict([image_filename], [xcenters], [ycenters])[0]
                for box_id, pos in enumerate(label):
                    rows.append(StageThreeRow(image_id, box_id, *pos))
                print(label)
        except KeyboardInterrupt:
            print('Stopping...')
        finally:
            if args.output_file:
                pandas.DataFrame(data=rows).to_csv(args.output_file)


main = RunStageThreeScript.run_main
