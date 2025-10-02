import torch 
import numpy as np 
import torch.nn.functional as F

def read_yuv_to_tensor(file_path, H, W, frame_num, in_format='444', out_format='444', interpolation='bilinear'):
    """
    讀取 yuv 檔案為 tensor,回傳 shape: (frame_num, C, H, W),C=3
    """
    class Read_YUV_Video():
        def __init__(self, file_path, in_format, out_format, H_W, frame_num, interpolation='bilinear'):
            self.files = open(file_path, "rb")
            self.in_format = in_format
            self.out_format = out_format
            self.H, self.W = H_W
            self.frame_num = frame_num
            self.interpolation = interpolation
            self.iter_cnt = 1
            self.y_size = self.H * self.W
            self.scale = 1 if in_format == '444' else 2
            self.uv_size = self.H * self.W * 2 if in_format == '444' else self.H * self.W // self.scale
        def read_one_frame(self):
            if self.iter_cnt > self.frame_num:
                raise Exception(f"Access frame out of range. Accessing {self.iter_cnt}th frame, but frame range set to {self.frame_num} !!!")
            self.iter_cnt += 1
            Y = self.files.read(self.y_size)
            UV = self.files.read(self.uv_size)
            Y = np.frombuffer(Y, dtype=np.uint8).reshape(1, self.H, self.W)
            UV = np.frombuffer(UV, dtype=np.uint8).reshape(2, self.H // self.scale, self.W // self.scale)
            Y = Y.astype(np.float32) / 255
            UV = UV.astype(np.float32) / 255
            Y = torch.from_numpy(Y).type(torch.FloatTensor)
            UV = torch.from_numpy(UV).type(torch.FloatTensor)
            if self.in_format == self.out_format:
                return Y, UV
            elif self.in_format == '420' and self.out_format == '444':
                UV = F.interpolate(UV.unsqueeze(0), scale_factor=2, mode=self.interpolation)[0]
                return Y, UV
            elif self.in_format == '444' and self.out_format == '420':
                UV = F.interpolate(UV.unsqueeze(0), scale_factor=1/2, mode=self.interpolation)[0]
                return Y, UV
            else:
                raise NotImplementedError

    def ycbcr444_to_rgb_Tensor(y, uv):
        y = y.unsqueeze(0)
        uv = uv.unsqueeze(0)
        y = y - (16/256)
        uv = uv - (128/256)
        yuv = torch.cat([y, uv], dim=1)
        T = torch.FloatTensor([[ 0.257,  0.504,   0.098],
                            [-0.148, -0.291,   0.439],
                            [ 0.439, -0.368,  -0.071]]).to(y.device)
        T = torch.linalg.inv(T)
        rgb = T.expand(yuv.size(0), -1, -1).bmm(yuv.flatten(2)).view_as(yuv)
        return rgb.clamp(min=0, max=1)[0]

    reader = Read_YUV_Video(file_path, in_format, out_format, (H, W), frame_num, interpolation)
    rgb_frames = []
    for _ in range(frame_num):
        y, uv = reader.read_one_frame()
        rgb = ycbcr444_to_rgb_Tensor(y, uv)
        rgb_frames.append(rgb)
    video_tensor = torch.stack(rgb_frames, dim=0)  # (T, 3, H, W)
    return video_tensor
