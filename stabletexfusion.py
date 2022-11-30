import os
import sys
import torch
import pytorch3d
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import math
import argparse
import glob
import PIL
from PIL import Image
from torch import autocast
import time
from pytorch_lightning import seed_everything
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.ops import roi_pool, masks_to_boxes
import dreamfusion

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    AmbientLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    HardFlatShader,
    BlendParams,
    TexturesUV,

)




def a_over_b(img_0, img_1):
    #composites A over B
    #expects [C,B,H], C = R,G,B,A
    alpha_0 = img_0[3,...]
    alpha_1 = img_1[3,...]
    alpha_out = alpha_0 + alpha_1 * (1 - alpha_0)
    C_out = img_0 * alpha_0 + img_1 * alpha_1 * (1 - alpha_0)
    C_out /= alpha_out 
    C_out = torch.nan_to_num(C_out)
    return C_out

def make_regions_of_interest(img, divisions=6, out_res=512):
    #expects [batch, H, W, C]
    #this is a cool failed test.....

    target_alpha = img[0:1,...,3] #[batch, H, W]
    alpha_shape = target_alpha.shape
    #this crops to the upper left
    num_grid_cells = divisions
    x_stride = alpha_shape[1]//num_grid_cells
    y_stride = x_stride
    target_alphas = []
    for i in range(0,num_grid_cells):
        for j in range(0,num_grid_cells):
            updated_alpha = target_alpha.clone()
            mask = torch.zeros_like(updated_alpha)
            mask[...,x_stride * i: x_stride * (i + 1), y_stride * j: y_stride * (j + 1)] = 1 #create a mask region
            updated_alpha *= mask

            if updated_alpha.sum() < 100: #remove small slivers or regions with no crops
                continue
            target_alphas.append(updated_alpha)


    target_alphas = torch.cat(target_alphas,dim=0)

    offset = 512 / 2
    boxes = masks_to_boxes(target_alphas)
    #ensure the boxes are all  512x512
    for i, box in enumerate(boxes):
        avg_x = (box[0] + box[2]) / 2.0
        avg_y = (box[1] + box[3]) / 2.0 
        boxes[i] = torch.tensor([avg_x-offset, avg_y-offset, avg_x + offset, avg_y + offset])


    boxes = F.pad(boxes,pad=(1,0),mode='constant', value=0)
    regions = roi_pool(img[0:1,...].permute(0,3,1,2),  boxes, out_res).permute(0,2,3,1) #crop boxes
    return regions

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)







def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

#given a uv map, sample a texture
def sample_texture(map_to_sample, uv_map):
    #uv_map expected to have a batch
    uv_remap = uv_map[...,0:2] #[B, H, W, C]
    uv_remap = uv_remap * 2.0 - 1.0
    uv_remap = torch.flip(uv_remap, [2])
    sample_texture_with_uv = F.grid_sample(map_to_sample.permute(0,3,1,2), uv_remap, align_corners=False).permute(0,2,3,1)
    sample_texture_with_uv = torch.flip(sample_texture_with_uv, [2])
    return sample_texture_with_uv

def plot_loss(losses):
    plt.close()
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l, label=  k + " loss")
    #ax.plot(loss, label="tex loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    return ax



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="artstation rock texture",
        help="the prompt to render"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="the version number for output paths",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=75,
        help="the guidance strength (should be high af)",
    )
    parser.add_argument(
        "--objpath",
        type=str,
        default="/objs/pighead.obj",
        help="the version number for output paths",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=6000,
        help="number of optimizer steps",
    )
    parser.add_argument(
        "--viz_every_n_images",
        type=int,
        default=600,
        help="output debug info every N images",
    )
    parser.add_argument(
        "--texsize",
        type=int,
        default=2048,
        help="output texture size (always square, so just use one number)",
    )

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    #stable diffusion stuff
    opt = parser.parse_args()
    seed_everything(opt.seed)


    #load mesh
    obj_filename = os.getcwd() + opt.objpath
    verts, faces, aux = load_obj(obj_filename)
    mesh = load_objs_as_meshes([obj_filename], device=device)

    #normalize input mesh
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    resolution = opt.texsize 

    #make a uv texture so we can sample the texture we're optimizing
    u = np.linspace(0,1,resolution)
    w = np.zeros((resolution,resolution))
    xv, yv = np.meshgrid(u,u)
    uvs = np.dstack((xv, yv, w))
    uvs = torch.tensor(uvs[None,...],dtype=torch.double).to(device) #[1,H,W,C]
    noise_tex = torch.randn(uvs.shape).to(device)
    tex = TexturesUV(maps=uvs.float(), faces_uvs=faces.textures_idx[None, ...].to(device),verts_uvs=aux.verts_uvs[None, ...].to(device)).to(device) 
    uv_mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex).to(device) #mesh with uv texture
    
    #normalize uv_mesh
    verts = uv_mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0]) / 1.0
    uv_mesh.offset_verts_(-center)
    uv_mesh.scale_verts_((1.0 / float(scale)))

    #we're gonna pre-render our uv textures :)
    #so set the number of cameras here
    num_views = 50
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-360, 360, num_views)


    lights = AmbientLights(device=device)
    pointlights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    pointlights2 = PointLights(device=device, location=[[0.0, 5.0, 0.0]])

    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=((.9, .9, 1.0),)) #, fov=30


    blend = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.,1.,1.))

    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
    )


    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras[0],
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras[0],
            lights=lights,
            blend_params=blend,
        )
    )

    #pre-render the renders of the mesh UV's we'll use in optimization step. cuz why make optimization time longer than it needs to be...
    target_images_uvs = []
    print("RENDERING UVS")
    for i in range(len(cameras)):
            with torch.no_grad():
                target_images_uvs.append(renderer(uv_mesh, cameras=cameras[i], lights=lights))
    target_images_uvs = torch.cat(target_images_uvs, dim=0)
    print("RENDERING COMPLETE")



    # The optimizer
    losses = {"rgb": {"weight": 1.0, "values": []}, }
    tex_shape = [1,resolution,resolution,3]
    tex_to_optimize_output = torch.zeros(tex_shape).to(device)
    tex_to_optimize_output.requires_grad_()
    tex_to_optimize = torch.rand(tex_shape).to(device)
    tex_to_optimize = tex_to_optimize.clamp(0,1)
    tex_to_optimize.requires_grad_()


    """
    The core idea here is we do the SDS loss which like optimizes a really ugly looking texture that kind of satisfies the idea:
    optimize for the texture, such that when it's encoded, and subsequently decoded, it matches the input text prompt
    """
    optimizer = torch.optim.AdamW([tex_to_optimize], lr=.1) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.95 ** max(iter // 150, 1))

    #second optimizer to do the like MSE loss pass
    optimizer_output = torch.optim.AdamW([tex_to_optimize_output], lr=.05)
    

    num_views_per_iteration = 1
    num_views_per_iteration = min(num_views, num_views_per_iteration)




    sd = dreamfusion.StableDiffusion(device)
    text_z = sd.get_text_embeds([opt.prompt], [""])

    print("Optimizing with prompt: {}".format(opt.prompt))
    Niter = opt.iters
    loop = tqdm(range(Niter))


    output_path_prefix = os.getcwd() + "/out3d/v{:04d}/".format(opt.version)
    tex_output_path = output_path_prefix + "output_textures/"
    render_output_path =output_path_prefix + "optim_images/" 
    latent_output_path =output_path_prefix + "latent_images/" 
    loss_output_path = output_path_prefix + "loss/" 


    os.makedirs(output_path_prefix, exist_ok=True)
    os.makedirs(tex_output_path, exist_ok=True)
    os.makedirs(render_output_path, exist_ok=True)
    os.makedirs(loss_output_path, exist_ok=True)
    os.makedirs(latent_output_path, exist_ok=True)


    tex_file_name = "textures_{:04d}.png" 
    render_file_name =  "render_{:04d}.png"
    loss_file_name = "LOSS_CHECKPT_{:04d}.png" 
    loss_RGB_file_name = "LOSS_RGB_CHECKPT_{:04d}.png" 
    latent_file_name = "latent_{:04d}.png"    
         
    loss_checkpts = Niter // opt.viz_every_n_images
    output_checkpts = Niter // opt.viz_every_n_images
    losses = {"SDS":[]}
    losses_output = {"RGB":[]}
    random_cam_list = np.random.permutation(num_views).tolist()



    affine_transfomer = torch.nn.Sequential(
        transforms.RandomAffine(degrees=(0, 0), translate=(.2, .2), scale=(1, 2)), #

    )

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

        loss = 0
        last_img = 0
        image_to_convert = []

        cam_lookup = i % num_views
        if cam_lookup == 0:
            random_cam_list = np.random.permutation(num_views).tolist()
        random_cam = random_cam_list[cam_lookup]

        uvs_to_sample = target_images_uvs[random_cam]


        #random augmentation, reminds me of the CLIP + VQGAN days :) 
        uvs_to_sample = affine_transfomer(uvs_to_sample.permute(2,0,1)).permute(1,2,0)
        alpha = uvs_to_sample[None,...,3:].permute(0,3,1,2)

        sampled_tex = sample_texture(tex_to_optimize, uvs_to_sample[None,...]).permute(0,3,1,2) #[1,C,H,W]
        sampled_tex = sampled_tex *  alpha + torch.zeros_like(sampled_tex) * (1 - alpha) #alpha compositing, i tried using rand instead of zeros at some point...
        


        loss = sd.train_step(text_z, sampled_tex , guidance_scale=opt.guidance)
        optimizer.step()
        scheduler.step()

        loss_rgb = 0
        
        #VISUALIZATION
        print("Loss SDS {0}: {1} \n Loss RGB {0}: {2}".format(i,loss.detach().mean(), loss_rgb))

        
        if i % (Niter // output_checkpts) == (Niter // output_checkpts) - 1:
            render_sample =  sampled_tex.detach().permute(0,2,3,1).cpu().clamp(0,1)[0].numpy()* 255   #[512,512,3]
            Image.fromarray(render_sample.astype(np.uint8)).save(render_output_path + render_file_name.format(i))
        
        losses["SDS"].append(loss.detach().mean().cpu())


        if i % (Niter // loss_checkpts) == (Niter // loss_checkpts) - 1:
            plot_loss(losses)
            plt.savefig(loss_output_path + loss_file_name.format(i))
            plt.close()
        
        if i % (Niter // loss_checkpts) == (Niter // loss_checkpts) - 1:

            
            latents = loss.detach().clone()
            # Img latents -> imgs
            imgs = sd.decode_latents(latents) # [1, 3, 512, 512]
            # Img to Numpy
            imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs * 255).round().astype('uint8')

            Image.fromarray(imgs[0]).save(latent_output_path + latent_file_name.format(i))

    Niter = 300
    loop_rgb_optim = tqdm(range(Niter)) 
    loss_checkpts = Niter // 100
    output_checkpts = Niter // 100    

    #to extract the final color mesh we need to do the same optimization we did previously, only using MSE loss against the decoded textures from various camera angles    
    for i in loop_rgb_optim:
        # Initialize optimizer
        optimizer_output.zero_grad()

        loss = 0
        last_img = 0
        image_to_convert = []

        batch_size = 1
        cam_lookup = i % (num_views // batch_size)
        if cam_lookup == 0:
            random_cam_list = np.random.permutation(num_views).tolist()
        random_cam = random_cam_list[cam_lookup]

        uvs_to_sample = target_images_uvs[random_cam]
        uvs_to_sample = affine_transfomer(uvs_to_sample.permute(2,0,1)).permute(1,2,0)

        alpha = uvs_to_sample[None,...,3:].permute(0,3,1,2)

        sampled_tex = sample_texture(tex_to_optimize, uvs_to_sample[None,...]).permute(0,3,1,2) #[1,3,512,512]
        sampled_tex = sampled_tex *  alpha + torch.zeros_like(sampled_tex) * (1 - alpha)

        #really great naming here, tex_to_optimize_output vs tex_to_optimize, surely not confusing for anyone....
        sampled_tex_output = sample_texture(tex_to_optimize_output, uvs_to_sample[None,...]).permute(0,3,1,2)


        pred_rgb_512 = F.interpolate(sampled_tex[:,0:3,...], (512, 512), mode='bilinear', align_corners=False)
        rgb_512_output = F.interpolate(sampled_tex_output[:,0:3,...], (512, 512), mode='bilinear', align_corners=False)

        latents = sd.encode_imgs(pred_rgb_512)
        imgs = sd.decode_latents(latents)
        loss = ((rgb_512_output - imgs) ** 2).mean()
        losses_output["RGB"].append(loss.detach().cpu())

        loss.backward()
        optimizer_output.step()


        if i % (Niter // output_checkpts) == (Niter // output_checkpts) - 1:
            output_tex =  tex_to_optimize_output.detach().cpu().clamp(0,1)[0].numpy()* 255                   
            Image.fromarray(output_tex.astype(np.uint8)).save(tex_output_path + tex_file_name.format(i))

        if i % (Niter // loss_checkpts) == (Niter // loss_checkpts) - 1:
            plot_loss(losses_output)
            plt.savefig(loss_output_path + loss_RGB_file_name.format(i))
            plt.close()

    pred_rgb_512 = tex_to_optimize_output.permute(0, 3, 1, 2)[:,0:3,...].detach().clone()
    imgs = pred_rgb_512.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    Image.fromarray(imgs[0]).save(output_path_prefix + "OUT_TEX.png")
    return 
    

if __name__ == "__main__":
    main()
