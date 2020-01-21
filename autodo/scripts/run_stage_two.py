from collections import namedtuple

import pandas
from os.path import join
from prettyparse import Usage

from autodo.scripts.base_script import BaseScript
from autodo.stage_two_predictor import StageTwoPredictor

StageTwoRow = namedtuple('StageTwoRow', 'image_id box_id yaw pitch roll center_x center_y')


class RunStageTwoScript(BaseScript):
    usage = Usage('''
        Train the stage two network
        
        :model_file str
            Model file to load from
        
        :boxes_file str
            Csv file with predicted boxes
        
        :cropped_folder str
            String with cropped images of boxes
        
        :-g --gpu
            Train with GPU
        
        :-o --output-file str -
            Output csv to write to
    ''')

    def run(self):
        args = self.args
        boxes = pandas.read_csv(args.boxes_file)
        predictor = StageTwoPredictor(args.model_file, args.gpu)
        rows = []
        try:
            for _, row in boxes.iterrows():
                image_id, box_id = row['image_id'], row['box_id']
                box = [row[i] for i in ['xmin', 'ymin', 'xmax', 'ymax']]
                image_filename = join(args.cropped_folder, '{}-{:02}.jpg'.format(image_id, box_id))
                label = predictor.predict([image_filename], [box])[0]
                rows.append(StageTwoRow(image_id, box_id, *label))
                print(label)
        except KeyboardInterrupt:
            print('Stopping...')
        finally:
            if args.output_file:
                pandas.DataFrame(data=rows).to_csv(args.output_file)


main = RunStageTwoScript.run_main
