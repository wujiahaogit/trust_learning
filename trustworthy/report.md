**Lime**


```python

def predict(input):
    input_tensor=torch.tensor(input).permute(0,3,1,2).float()
    # 将 input 转换成PyTorch张量，并使用permute()函数将张量的维度重新排列。 input:(batches, height, width, channels)->(batches, channels,height, width)
    output = model(input_tensor)
    # 使用model进行预测
    return output.detach().numpy()
    # 返回output
```

```python
for idx, (image,label) in enumerate(zip(all_image,real_label)):
    x = (image/255).astype(np.double)

    # Refer the doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance
    ###################################
    # write the code here
    explaination = lime_image.LimeImageExplainer().explain_instance(x, predict, segmentation_fn=segmentation)
    # 创建一个LimeImageExplain对象，然后调用它的explain_instance方法来解释模型的预测结果
    ###################################


    # Turn the result from explainer to the image
    # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask
    lime_img, mask = explaination.get_image_and_mask(label=label.item(),positive_only=False,hide_rest=False,num_features=11,min_weight=0.05)
    axs[idx].imshow(lime_img)
    axs[idx].set_xticks([])
    axs[idx].set_yticks([])
```

**Saliency Map**

```python
def compute_saliency_maps(x, y, model):
  # input: the input image, the ground truth label, the model
  # output: the saliency maps of the images
  # We need to normalize each image, because their gradients might vary in scale, use the "normalize" function.
  # pass
  ###################################
  # write the code here
  saliencies = []
  for image, label in zip(x, y):
      image = image.unsqueeze(0)
      # 添加一个批处理维度
      image.requires_grad_()
      # 设置图像的requires_grad_()为True，从而可以反向传播时计算梯度

      model.zero_grad()
      # 模型梯度归零
      output = model(image)
      # 进行预测
      score = output[0, label.item()]
      # 获取与标签对应的输出分数
      score.backward()
      # 反向传播

      saliency = image.grad.abs().squeeze().max(dim=0)[0]
      # 计算图像梯度的绝对值，并沿着通道维度取最大值
      saliencies.append(saliency)
      # 将saliency加入到saliencies中
    
  saliencies = [normalize(saliency) for saliency in saliencies]
  # 对saliencies中的元素进行nomalize
  return saliencies
  # return saliencies
  ###################################
```

**Smooth Grad**

```python
def smooth_grad(x, y, model, epoch, param_sigma_multiplier):
  # input: the input image, the ground truth label, the model, the number for average, Sigma multiplier when calculating std of noise
  # output: the saliency maps of the images
  smooth_saliencies = []
  # 循环epoch次，在每次迭代中添加随机噪声到输入图像上，并裁剪到[0,1]的范围内
  for _ in range(epoch):
      noise = torch.randn_like(x) * param_sigma_multiplier
      # 噪声

      noisy_image = x + noise
      # 添加到输入图像上

      noisy_image = torch.clamp(noisy_image, 0, 1)
      # 裁剪到[0,1]上，避免出现溢出

      noisy_image.requires_grad_()
      # 设置图像的requires_grad_()为True，从而可以反向传播时计算梯度
      
      model.zero_grad()
      # 模型梯度归零

      output = model(noisy_image)
      # 进行预测

      score = output[0, y.item()]
      # 获取与标签对应的输出分数

      score.backward()
      # 反向传播

      saliency = noisy_image.grad.abs().squeeze()
      # 计算噪声添加后图像的梯度的绝对值

      smooth_saliencies.append(saliency)
      # 添加到smooth_saliencies中
    
  smooth_saliencies = torch.stack(smooth_saliencies).mean(dim=0)
  # 将每个梯度张量堆叠起来，然后对epoch维度求平均值

  smooth_saliencies = [normalize(saliency) for saliency in smooth_saliencies]
  # 进行nomalize

  return smooth_saliencies
```

**CAM**

```python
def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    ###################################
    # write the code here
    # hint: the cam_img should be integers in [0,255] so that it can be transffered by cv2.applyColorMap later.
    cam = np.matmul(weight_softmax[class_idx], feature_conv.reshape((nc, h*w)))
    # 将特征图与对应类别的softmax权重进行点积，得到每个位置的激活强度

    cam = cam.reshape(h, w)
    # 将一维的CAM reshape为二维
    
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    # 归一化为[0,1]的范围内
    
    cam_img = np.uint8(255 * cam_img)
    # 转化为[0,255]之间的unit8整数
    return cv2.resize(cam_img, size_upsample)
```


**Paper reading**

[9] SmoothGrad: removing noise by adding noise.

一. The brief summary:

The paper introduces SMOOTHGRAD, a simple method that can help visually sharpen gradient-based sensitivity maps, and it compares the SMOOTHGRAD method to several gradient-based sensitivity map methods and demonstrates its effects. The authors provide a conjecture why it might be more reflective of how the network is doing classification. It also discusses combining SmoothGrad with other methods to improve the visual coherence of sensitivity maps, and shows the broad prospect of SMOOTHGRAD.

二. The strength:

1. It suggests a new and creative approach
2. It compares the SMOOTHGRAD method to several gradient-based sensitivity map methods and demonstrates its effects,shows a side-by-side comparison between SMOOTHGRAD and three gradient-based methods.
3. Its shows effect of noise level and sample size by experiments.
4. It trys to combines SmoothGrad with other methods and add noise during training. Both approaches have had good results.

三. The weakness:

1. It provides that noisy sensitivity maps are due to noisy gradients, but there is no solid theoretical basis for this argument.
2. Whether SMOOTHGRAD method of de-noising is applicable to other areas of engineering has not been explored.
3.  Is shows that applying 10%-20% noise seems to balance the sharpness of sensitivity map and maintain the structure of the original image, but doesn't give the theoretical basis.