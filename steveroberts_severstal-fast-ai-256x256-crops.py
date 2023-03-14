#!/usr/bin/env python
# coding: utf-8

# In[1]:


# when set true, the index is removed from the negative tiles (this is already done for +ve tiles)
# - as a result no -ve images are ever added to the validation set
# - also positive and negative tiles can come from the same original images - with old code tiles
#   from same image could end up both training and validation sets, resulting in data leakage
USE_ORIGINAL_VALIDATION = True


# In[2]:


fold = 0


# In[3]:


NotebookTitle = "Steel Defect Segmentation - Training - Original Validation"


# In[4]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import gc
import os
from sklearn.model_selection import KFold
from PIL import Image
import zipfile
import io
import cv2
import warnings
from radam import RAdam
warnings.filterwarnings("ignore")

fastai.__version__


# In[5]:


sz = 256
bs = 16
nfolds = 4

SEED = 2019
TRAIN = '../input/severstal-256x256-images-with-defects/images/'
MASKS = '../input/severstal-256x256-images-with-defects/masks/'
TRAIN_N = '../input/severstal-256x256-images-with-defects/images_n/'
HARD_NEGATIVE = '../input/hard-negative-severstal-crops/pred.csv'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #tf.set_random_seed(seed)
seed_everything(SEED)
torch.backends.cudnn.benchmark = True


# In[6]:


#the code below modifies fast.ai functions to incorporate Hcolumns into fast.ai Dynamic Unet

from fastai.vision.learner import create_head, cnn_config, num_features_model, create_head
from fastai.callbacks.hooks import model_sizes, hook_outputs, dummy_eval, Hook, _hook_inner
from fastai.vision.models.unet import _get_sfs_idxs, UnetBlock

class Hcolumns(nn.Module):
    def __init__(self, hooks:Collection[Hook], nc:Collection[int]=None):
        super(Hcolumns,self).__init__()
        self.hooks = hooks
        self.n = len(self.hooks)
        self.factorization = None 
        if nc is not None:
            self.factorization = nn.ModuleList()
            for i in range(self.n):
                self.factorization.append(nn.Sequential(
                    conv2d(nc[i],nc[-1],3,padding=1,bias=True),
                    conv2d(nc[-1],nc[-1],3,padding=1,bias=True)))
                #self.factorization.append(conv2d(nc[i],nc[-1],3,padding=1,bias=True))
        
    def forward(self, x:Tensor):
        n = len(self.hooks)
        out = [F.interpolate(self.hooks[i].stored if self.factorization is None
            else self.factorization[i](self.hooks[i].stored), scale_factor=2**(self.n-i),
            mode='bilinear',align_corners=False) for i in range(self.n)] + [x]
        return torch.cat(out, dim=1)

class DynamicUnet_Hcolumns(SequentialEx):
    "Create a U-Net from a given architecture."
    def __init__(self, encoder:nn.Module, n_classes:int, blur:bool=False, blur_final=True, 
                 self_attention:bool=False,
                 y_range:Optional[Tuple[float,float]]=None,
                 last_cross:bool=True, bottle:bool=False, **kwargs):
        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni*2, **kwargs),
                                    conv_layer(ni*2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]
        hc_c = [x.shape[1]]
        
        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sfs_idxs)-3)
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, 
                blur=blur, self_attention=sa, **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)
            self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
            hc_c.append(x.shape[1])

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        hc_c.append(ni)
        layers.append(Hcolumns(self.hc_hooks, hc_c))
        layers += [conv_layer(ni*len(hc_c), n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()
            
def unet_learner(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
        norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, 
        blur:bool=False, self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, 
        last_cross:bool=True, bottle:bool=False, cut:Union[int,Callable]=None, 
        hypercolumns=True, **learn_kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    M = DynamicUnet_Hcolumns if hypercolumns else DynamicUnet
    model = to_device(M(body, n_classes=data.c, blur=blur, blur_final=blur_final,
        self_attention=self_attention, y_range=y_range, norm_type=norm_type, 
        last_cross=last_cross, bottle=bottle), data.device)
    learn = Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn


# In[7]:


def dice(input:Tensor, targs:Tensor, iou:bool=False, eps:float=1e-8)->Rank0Tensor:
    n,c = targs.shape[0], input.shape[1]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect,union = [],[]
    for i in range(1,c):
        intersect.append(((input==i) & (targs==i)).sum(-1).float())
        union.append(((input==i).sum(-1) + (targs==i).sum(-1)).float())
    intersect = torch.stack(intersect)
    union = torch.stack(union)
    if not iou: return ((2.0*intersect + eps) / (union+eps)).mean()
    else: return ((intersect + eps) / (union - intersect + eps)).mean()


# In[8]:


class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
    
class SegmentationItemList(SegmentationItemList):
    _label_cls = SegmentationLabelList

# Setting transformations on masks to False on test set
def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
    if not tfms: tfms=(None,None)
    assert is_listy(tfms) and len(tfms) == 2
    self.train.transform(tfms[0], **kwargs)
    self.valid.transform(tfms[1], **kwargs)
    kwargs['tfm_y'] = False # Test data has no labels
    if self.test: self.test.transform(tfms[1], **kwargs)
    return self
fastai.data_block.ItemLists.transform = transform

def open_mask(fn:PathOrStr, div:bool=True, convert_mode:str='L', cls:type=ImageSegment,
        after_open:Callable=None)->ImageSegment:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        #generate empty mask if file doesn't exist
        x = PIL.Image.open(fn).convert(convert_mode)           if Path(fn).exists()           else PIL.Image.fromarray(np.zeros((sz,sz)).astype(np.uint8))
    if after_open: x = after_open(x)
    x = pil2tensor(x,np.float32)
    return cls(x)


# In[9]:


df = pd.read_csv(HARD_NEGATIVE)
df['index'] = df.index
df.plot(x='index', y='pixels', kind = 'line');
plt.yscale('log')


# In[10]:


df.head()


# In[11]:


# To fix problem with Fast.ai display of tiled images
import matplotlib
cmap = matplotlib.colors.ListedColormap(['grey', 'blue', 'red', 'green', 'yellow'])


# In[12]:


stats = ([0.396,0.396,0.396], [0.179,0.179,0.179])
#check https://www.kaggle.com/iafoss/256x256-images-with-defects for stats


# In[13]:


# create a set of image names (without tile indexes) e.g. 'b69aaf096'
img_p = set([p.stem[:-2] for p in Path(TRAIN).ls()])

#select 12000 of the most difficult negative exaples
neg = list(pd.read_csv(HARD_NEGATIVE).head(12000).fname)
neg = [Path(TRAIN_N)/f for f in neg]

# set of negative tiles - these have retained their indexes?
# - does the hard negative list only contain one tile per image?

# set of negative tiles 
# - for some reason these have retained their indexes
# - as a result the set 'img_n' contains multiple entries from the same image
# - keeping the indexes len(img_n) = 12000 (i.e. the full set)
# - removing the indexes len(img_n) = 6732

if USE_ORIGINAL_VALIDATION:
    img_n = set([p.stem for p in neg])
else:
    img_n = set([p.stem[:-2] for p in neg])

print("Negative Set:",len(img_n)," (out of 12000)")

# create a set that contains entries that are in either the positive OR the negative sets
# - i.e. duplicates will be dropped 
# (duplicates are possible since negative set can contain tiles from parts of a defect image where the mask is empty)
# - however, since the indexes are retained on negative tiles, the names will never match
img_set = img_p | img_n

# sorting the sets converts them back to lists
img_p_list = sorted(img_p)
img_n_list = sorted(img_n)

# combine the list of positive and negative images
# - when indexes haven't been removed from negative, this list will be identical to the OR'd set
# - when indexes HAVE been removed it will be different, since duplicates between +ve and -ve may exist
img_list = img_p_list + img_n_list

# MY CODE - use the set, which removes duplicates from the +ve and -ve sets, rather than ANDed
if USE_ORIGINAL_VALIDATION == False:
    img_list = sorted(img_set)

print("OR'd Set: ",len(img_set))
print("AND'd Set: ",len(img_list))


# In[14]:


# where nfolds=4
kf = KFold(n_splits=nfolds, shuffle=True, random_state=SEED)


# In[15]:


# valid_idx = list(kf.split(list(range(len(img_list)))))[fold][1]

# the length of the list of +ve and -ve
# - so for positive is the image names
# - for negative is the tile names
# = 19260 
len_img_list = len(img_list)

# creates a list of all indexes from 0 to len_img_list
list_img_list = list(range(len_img_list))

# create a list of indexes for the 4 folds
fold_lists = list(kf.split(list_img_list))

# get the indexes for the first fold - this contains 2 arrays - the training indexes and the validation indexes
fold_list_0 = fold_lists[fold]

# get the indexes of the first fold's validation set
valid_idx = fold_list_0[1]
print(f'Length of validation indexes = {len(valid_idx)}')

# the set of tile names\image names for this fold's validation set
valid = set([img_list[i] for i in valid_idx])

print(f'Validation Set for fold 0: {list(valid)[:10]}')

# now compare the stem names of all images in the positive and negative sets
# against the names in the validation set
# - since the negative images haven't had their index removed in the validation set
# they will never match with the stem names and so none will be added to these indexes
# - the size of this list is larger than the size of the valid indexes above because all positive
# tiles in the valid list are added
# - the only tiles from the negative set that are added are ones from images that also had defect tiles (i.e. in the positive set)
valid_idx = []
for i,p in enumerate(Path(TRAIN).ls() + neg):
    if p.stem[:-2] in valid: 
        valid_idx.append(i)

        
print(f'Length of validation indexes after stem compare: {len(valid_idx)}')    

# tile image '7e0a4584c_0' is from the negative set and is one of the images for fold 0's validation set
# - therefore all '7e0a4584c' images should be added to the validation set
# - however, since no match will occur in the 'if' statement above, no indexes from negative set are added

# Create databunch
sl = SegmentationItemList.from_folder(TRAIN)

print(f'Number of positive images: {len(Path(TRAIN).ls())}')
print(f'Number of negative images: {len(neg)}')

# add the negative images to the item list
sl.items = np.array((list(sl.items) + neg))

# the total size is all positive and negative tiles = 31695
print(f'Total size of all data in item list: {sl.items.shape}')

data = (sl.split_by_idx(valid_idx)
    .label_from_func(lambda x : str(x).replace('/images', '/masks'), classes=[0,1,2,3,4])
    .transform(get_transforms(xtra_tfms=dihedral()), size=sz, tfm_y=True)
    .databunch(path=Path('.'), bs=bs)
    .normalize(stats))


# In[16]:


# show the train\validation split - with 4 folds this should be about 3:1
# - with old code = 25644 : 6051 = 4:1
# - with new code = 23667 : 8028 = 3:1
data


# In[17]:


def get_data(fold):
    #split with making sure that crops of the same original image 
    #are not shared between folds, so additional training and validation 
    #could be done on full images later
    
    # get the validation indexes of the specified fold
    valid_idx = list(kf.split(list(range(len(img_list)))))[fold][1]
    
    # get the set of tile names\image names for this fold's validation set
    # - because indexes weren't removed from negative tiles these are still present
    # in the names added to this validation set
    valid = set([img_list[i] for i in valid_idx])
    
    # now removes the index from all tiles in both the positive and negative sets
    # - however, since the negative tiles haven't had their index removed in the validation set
    # their indexes will never be added to valid_idx
    valid_idx = []
    for i,p in enumerate(Path(TRAIN).ls() + neg):
        if p.stem[:-2] in valid: valid_idx.append(i)
            
    # Create databunch
    sl = SegmentationItemList.from_folder(TRAIN)
    sl.items = np.array((list(sl.items) + neg))
    data = (sl.split_by_idx(valid_idx)
        .label_from_func(lambda x : str(x).replace('/images', '/masks'), classes=[0,1,2,3,4])
        .transform(get_transforms(xtra_tfms=dihedral()), size=sz, tfm_y=True)
        .databunch(path=Path('.'), bs=bs)
        .normalize(stats))
    return data

# Display some images with masks - set the cmap to see the different defects
# NOTE:- with USE_ORIGINAL_VALIDATION = False there will be more validation images without defects
# since images from the negative set are also now included
get_data(0).show_batch(cmap=cmap,vmax=5)
# get_data(0).show_batch()


# In[18]:


@dataclass
class CSVLogger(LearnerCallback):
    def __init__(self, learn, filename= 'history'):
        self.learn = learn
        self.path = self.learn.path/f'{filename}.csv'
        self.file = None

    @property
    def header(self):
        return self.learn.recorder.names

    def read_logged_file(self):
        return pd.read_csv(self.path)

    def on_train_begin(self, metrics_names: StrList, **kwargs: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        e = self.path.exists()
        self.file = self.path.open('a')
        if not e: self.file.write(','.join(self.header) + '\n')

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        self.write_stats([epoch, smooth_loss] + last_metrics)

    def on_train_end(self, **kwargs: Any) -> None:
        self.file.flush()
        self.file.close()

    def write_stats(self, stats: TensorOrNumList) -> None:
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.header, stats)]
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')


# In[19]:


# Prediction with flip TTA
def model_pred(learn:fastai.basic_train.Learner,F_save,
        ds_type:fastai.basic_data.DatasetType=DatasetType.Valid, 
        tta:bool=True): #if use train dl, disable shuffling
    learn.model.eval();
    dl = learn.data.dl(ds_type)
    #sampler = dl.batch_sampler.sampler
    #dl.batch_sampler.sampler = torch.utils.data.sampler.SequentialSampler(sampler.data_source)
    name_list = [Path(n).stem for n in dl.dataset.items]
    num_batchs = len(dl)
    t = progress_bar(iter(dl), leave=False, total=num_batchs)
    count = 0
    with torch.no_grad():
        for x,y in t:
            x = x.cuda()
            py = torch.softmax(learn.model(x),dim=1).permute(0,2,3,1).detach()
            if tta:
                flips = [[-1],[-2],[-2,-1]]
                for f in flips:
                    py += torch.softmax(torch.flip(learn.model(torch.flip(x,f)),f),dim=1)                      .permute(0,2,3,1).detach()
                py /= (1+len(flips))
            py = py.cpu().numpy()
            batch_size = len(py)
            for i in range(batch_size):
                taget = y[i].detach().cpu().numpy() if y is not None else None
                F_save(py[i],taget,name_list[count])
                count += 1
    #dl.batch_sampler.sampler = sampler
    
def save_img(data,name,out):
    data = data[:,:,1:]
    img = cv2.imencode('.png',(data*255).astype(np.uint8))[1]
    out.writestr(name, img)
    
#dice for threshold selection
def dice_np(pred, targs, noise_th = 0, eps=1e-7):
    targs = targs[0,:,:]
    c = pred.shape[-1]
    pred = np.argmax(pred, axis=-1)
    dices = []
    for i in range(1,c):
        if (pred==i).sum() > noise_th:
            intersect = ((pred==i) & (targs==i)).sum().astype(np.float)
            union = ((pred==i).sum() + (targs==i).sum()).astype(np.float)
            dices.append((2.0*intersect + eps) / (union + eps))
        else: dices.append( 1.0 if (targs==i).sum() == 0 else 0.0)
    return np.array(dices).mean()


# In[20]:


# get rid of garbage before training starts
gc.collect()


# In[21]:


dices = []
noise_ths = np.arange(0, 1501, 125)

with zipfile.ZipFile('val_masks_tta.zip', 'w') as archive_out:
    
    #the function to save val masks and dices
    def to_mask(yp, y, id):
        name = id + '.png'
        save_img(yp,name,archive_out)
        dices_th = []
        for noise_th in noise_ths:
            dices_th.append(dice_np(yp,y,noise_th))
        dices.append(dices_th)
        
    
    data = get_data(fold)
    learn = unet_learner(data, models.resnet34, metrics=[dice], opt_func=RAdam)
    learn.clip_grad(1.0);
    logger = CSVLogger(learn,f'log{fold}')

    #fit the decoder part of the model keeping the encode frozen
    lr = 1e-3
    learn.fit_one_cycle(4, lr, callbacks = [logger])
    
    #fit entire model with saving on the best epoch
    learn.unfreeze()
    learn.fit_one_cycle(15, slice(lr/50, lr/2), callbacks = [logger])
        
    #save model
    learn.save('fold'+str(fold));
    np.save('items_fold'+str(fold), data.valid_ds.items)
     
    #run TTA prediction on val, save masks and dices
    model_pred(learn,to_mask)
    
    gc.collect()
    torch.cuda.empty_cache()
dices = np.array(dices).mean(0)


# In[22]:


best_dice = dices.max()
best_thr = noise_ths[dices.argmax()]
plt.figure(figsize=(8,4))
plt.plot(noise_ths, dices)
plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
plt.text(best_thr+50, best_dice-0.01, f'DICE = {best_dice:.3f}', fontsize=14);
plt.show()

