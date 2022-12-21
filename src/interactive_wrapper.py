from skimage.io import imread, imsave
import pims
import os
import time

def image_flash_removal(ambient_path: str, flash_path: str, out_path: str,
    sig: float, thld: float, bound_cond = 2, init_opt = 2,\
        conv=5e-4, niter=1000):
    overall_start = time.time()
    os.system(f'src/gradient_domain_gpu {ambient_path} {flash_path} {out_path} 1 {sig} {thld} {bound_cond} {init_opt} {conv} {niter}')
    print(f'Image fusion processed in {time.time() - overall_start} seconds.\nGet the result in {out_path}.')


def video_poisson_fusion(video_path: str, patch_path: str,
    row : int, col: int, conv=5e-4, niter=1000):
    '''
    Patch a small picture seamlessly to every frame of the video.
    Output the process duration for each frame,
    and final execution time on all frames.
    '''
    patch = imread(patch_path)[:, :, :3]
    nrow, ncol = patch.shape[:-1]
    jpg_patch_path = '.'.join(patch_path.split('.')[:-1]) + '.jpg'
    imsave(jpg_patch_path, patch, quality=85)

    overall_start = time.time()

    frames = pims.Video(video_path)
    for idx in range(len(frames)):
        st = time.time()
        imsave('/tmp/temp_input.jpg', frames[idx][row:row+nrow, col:col+ncol], quality=85)
        # print(f'Frame {idx} process time:', time.time() - st)
        os.system(f'src/gradient_domain_gpu /tmp/temp_input.jpg {patch_path} /tmp/temp_output.jpg 2 {conv} {niter}')
        # print(f'Frame {idx} process time:', time.time() - st)
        frames[idx][row:row+nrow, col:col+ncol] = imread('/tmp/temp_output.jpg')
        imsave(f'/tmp/frame{idx:06}.jpg', frames[idx], quality=85)
        # print(f'Frame {idx} process time:', time.time() - st)

    os.system('ffmpeg -framerate 30 -pattern_type glob -i \'/tmp/frame*.jpg\' -c:v libx264 -pix_fmt yuv420p out.mp4')
    print(f'{len(frames)} frames processed in {time.time() - overall_start} seconds.\nFPS {len(frames) / (time.time() - overall_start)} \nGet the result in out.mp4.')

    

if __name__ == '__main__':
    video_poisson_fusion('demo_video.mp4', '/tmp/patch.png', 50, 300)
    