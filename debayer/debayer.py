import torch
import torch.nn
import torch.nn.functional

class Debayer3x3(torch.nn.Module):
    '''Demosaicing of Bayer images using 3x3 convolutions.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.
    
    Kwargs
    ------
    shape : tuple (optional)
        height and width of image. If omitted, shape is derived
        from actual image. When image size does not change, setting
        shape increases performance.
    '''

    def __init__(self, shape=None):
        super(Debayer3x3, self).__init__()

        if shape is None:
            shape=(2,2)

        self.kernels = torch.nn.Parameter(
            torch.tensor([
                [0,0,0],
                [0,1,0],
                [0,0,0],
                
                [0, 0.25, 0],
                [0.25, 0, 0.25],
                [0, 0.25, 0],
                
                [0.25, 0, 0.25],
                [0, 0, 0],
                [0.25, 0, 0.25],
                
                [0, 0, 0],
                [0.5, 0, 0.5],
                [0, 0, 0],
                
                [0, 0.5, 0],
                [0, 0, 0],
                [0, 0.5, 0],
            ]).view(5,1,3,3), requires_grad=False
        )

        
        self.index = torch.nn.Parameter(
            torch.tensor([
                # dest channel r
                [0, 3], # pixel is R,G1
                [4, 2], # pixel is G2,B
                # dest channel g
                [1, 0], # pixel is R,G1
                [0, 1], # pixel is G2,B
                # dest channel b
                [2, 4], # pixel is R,G1
                [3, 0], # pixel is G2,B
            ]).view(1,3,2,2).repeat(1,1,shape[0]//2,shape[1]//2), requires_grad=False
        )
        

    @torch.no_grad()
    def forward(self, x):
        '''Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        '''

        x = torch.nn.functional.pad(x, (1,1,1,1), mode='replicate')
        c = torch.nn.functional.conv2d(x, self.kernels, stride=1)
        idx = self.index
        if idx.shape[-2:] != c.shape[-2:]:
            idx = idx.repeat(1,1,c.shape[-2]//2,c.shape[-1]//2)
        rgb = torch.gather(c, 1, idx)
        return rgb

class Debayer2x2(torch.nn.Module):
    '''Demosaicing of Bayer images using 2x2 convolutions.
    
    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.        
    '''

    def __init__(self, **kwargs):
        super(Debayer2x2, self).__init__()

        self.kernels = torch.nn.Parameter(
            torch.tensor([
                [1, 0],
                [0, 0],

                [0, 0.5],
                [0.5, 0],

                [0, 0],
                [0, 1],
            ]).view(3,1,2,2), requires_grad=False
        )

    @torch.no_grad()
    def forward(self, x):
        '''Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        '''

        x = torch.nn.functional.conv2d(x, self.kernels, stride=2)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x