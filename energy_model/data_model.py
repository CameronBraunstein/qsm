
from utils.file_io.dataloading.dataloader import get_dataloader

def get_data_model(args):
    dm = DataModel(args)
    return dm

class DataModel:
    def __init__(self,args):
        self.dataloader = get_dataloader(args)
    
    def initialize_model_at_step(self,step):
        self.frame_0 = self.dataloader.get_frame(0,step=step,convert_to_gray_scale=True)
        self.frame_1 = self.dataloader.get_frame(1,step=step,convert_to_gray_scale=True)
        self.max_width = self.frame_0.shape[1]-1

    def get_data_energy(self,base_y,base_x,displacement):
        pixel_value_0 = self._get_pixel_value(0,base_y,base_x)
        pixel_value_1 = self._get_pixel_value(1,base_y,base_x,displacement=displacement)
        squared_diff = (pixel_value_0 - pixel_value_1)**2
        return squared_diff


    def _get_pixel_value(self,frame_number,y,x,displacement=0):
        if frame_number == 0:
            frame = self.frame_0
        elif frame_number == 1:
            x += displacement
            x = min(x,self.max_width)
            x = int(x)
            frame = self.frame_1
        pixel_value = frame[y,x]
        return pixel_value
        




    
