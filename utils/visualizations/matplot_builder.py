
#TODO: DEPRECATE FILE

from utils.file_io.dataloading.dataloader import get_dataloader
import matplotlib.pyplot as plt

def get_matplotlib_builder(args):
    mplb = MatplotlibBuilder(args)
    return mplb

def _bw_to_blue_cyan(values,name):
    plt.axis('off')
    plt.imshow(values, cmap='gray', vmin=0, vmax=255)

    plt.savefig("/home/cbraunst/Desktop/{}.png".format(name),bbox_inches='tight')
    plt.show()
    return None

# def crop_tsukuba(displacements):
#         top_offset = 18
#         bottom_offset = 18
#         left_offset = 18
#         right_offset = 18
#         displacements = displacements[top_offset:-bottom_offset, left_offset:-right_offset]
#         return displacements

class MatplotlibBuilder:
    def __init__(self,args):
        self.dataloader = get_dataloader(args)
        self.scene_name = args.scene_name
        self.scene_frame_number = args.scene_frame_number

    def make_custom_png(self):
        image = self.dataloader.get_estimated_displacements(step=5,can_post_process=True,for_display=False,for_evaluation=True)
        #image = self.dataloader.get_ground_truth_displacements(step=5,for_display=True,for_evaluation=True)
        image*=255
        scale_factor = 4 # x1 for gray x4 for jet
        cmap_scheme = 'jet' #gray or jet
        #image = np.clip(image,None, 256//scale_factor-1)
        image = image * scale_factor
        image = 255 - image
        plt.axis('off')
        
        plt.imshow(image, cmap=cmap_scheme, vmin=0, vmax=255)
        plt.savefig("tmp/{}_{}.png".format(self.scene_name,self.scene_frame_number),bbox_inches='tight')


    def make_fig_png(self):
        #image = self.dataloader.get_estimated_displacements(step=5,can_post_process=True,for_display=True,for_evaluation=True)
        #image = self.dataloader.get_ground_truth_displacements(step=5,for_display=True,for_evaluation=False)
        image = self.dataloader.get_frame(0,step=2,convert_to_gray_scale=False)
        image = self.dataloader._crop_for_evaluation(image)
        #image = image[9:-9,9:-9]
        #image = image[5:-5,5:-5]
        

        plt.axis('off')
        plt.imshow(image, vmin=0, vmax=1)
        plt.savefig("/home/cbraunst/Desktop/{}.png".format(self.scene_name),bbox_inches='tight')
        plt.show()

        raise Exception('STOP')

        image*=255
        scale_factor = 1 # x1 for gray x4 for jet
        cmap_scheme = 'gray' #gray or jet
        self._make_and_save_fig(image,scale_factor,cmap_scheme)


    def _make_and_save_fig(self,image,scale_factor,cmap_scheme):
        #image = np.clip(image,None, 256//scale_factor-1)
        image = image * scale_factor
        plt.axis('off')
        plt.imshow(image, cmap=cmap_scheme, vmin=0, vmax=255)
        plt.savefig("/home/cbraunst/Desktop/{}.png".format(self.scene_name),bbox_inches='tight')
        plt.show()