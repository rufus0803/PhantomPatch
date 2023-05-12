from patch_config import *
from load_data import *
import load_data
import gc
import matplotlib.pyplot as plt
from torch import autograd
plt.rcParams["axes.grid"] = False
plt.axis('off')
# from pytorch.yolo_models.darknet import Darknet

if __name__ == "__main__":

    img_dir = "inria/Train/pos"
    lab_dir = "inria/Train/yolo-labels"
    cfgfile = "cfg/yolo.cfg"
    weightfile = "weights/yolov3.weights"
    printfile = "non_printability/30values.txt"
    patch_size = 300

    print('LOADING MODELS')
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)

    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    prob_extractor = MaxProbExtractor(0, 80).cuda()

    nps_calculator = NPSCalculator(printfile, patch_size)
    nps_calculator = nps_calculator.cuda()
    total_variation = TotalVariation().cuda()
    print('MODELS LOADED')

    img_size = darknet_model.height
    batch_size = 1#10#18
    n_epochs = 20
    max_lab = 14

    # Choose between initializing with gray or random
    adv_patch_cpu = torch.full((3,patch_size,patch_size),0.5)
    #adv_patch_cpu = torch.rand((3,patch_size,patch_size))


    patch_img = Image.open("cup.jpg").convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)

    adv_patch_cpu.requires_grad_(True)

    train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, max_lab, img_size, shuffle=True),
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=10)

    optimizer = optim.Adam([adv_patch_cpu], lr=.0001, amsgrad=True)

    #try:
    et0 = time.time()
    for epoch in range(n_epochs):
        ep_det_loss = 0
        det_loss = 1e-6
        bt0 = time.time()
        for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
            with autograd.detect_anomaly():
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                adv_patch = adv_patch_cpu.cuda()
                adv_batch_t = patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True)
                p_img_batch = patch_applier(img_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))

                output = darknet_model(p_img_batch)

                max_prob = prob_extractor(output)

                nps = nps_calculator(adv_patch)
                tv = total_variation(adv_patch)

                det_loss = torch.mean(max_prob)
                ep_det_loss += det_loss.detach().cpu().numpy()

                nps_loss = nps*0.01
                tv_loss = tv*2.5
                loss = det_loss + nps_loss + tv_loss

                if loss.isnan():
                    loss = 1e-6
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        # print('BATCH', i_batch, end='...\n')
                        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                        # plt.imshow(im)
                        # plt.show()

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                bt0 = time.time()
        et1 = time.time()
        ep_det_loss = ep_det_loss/len(train_loader)
        # ep_det_loss = ep_det_loss
        ep_nps_loss = nps_loss.detach().cpu().numpy()
        ep_tv_loss = tv_loss.detach().cpu().numpy()
        tot_ep_loss = ep_det_loss + ep_nps_loss + ep_tv_loss

        if True:
            print('  EPOCH NR: ', epoch),
            print('EPOCH LOSS: ', tot_ep_loss)
            print('  DET LOSS: ', ep_det_loss)
            print('  NPS LOSS: ', ep_nps_loss)
            print('   TV LOSS: ', ep_tv_loss)
            print('EPOCH TIME: ', et1-et0)
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.show()
            im.save("patch.png")
            del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
            torch.cuda.empty_cache()
        et0 = time.time()


    from load_data import *
    import load_data

    import matplotlib.pyplot as plt
    plt.rcParams["axes.grid"] = False

    def pad_and_scale(img, lab, imgsize):
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        padded_img = padded_img.resize((imgsize,imgsize))     #choose here
        return padded_img, lab


    img_path = "test/img/crop001024.png"
    lab_path = "test/lab/crop001024.txt"

    cfgfile = "cfg/yolo.cfg"
    weightfile = "weights/yolov3.weights"

    printfile = "non_printability/30values.txt"
    patch_size = 300

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    prob_extractor = MaxProbExtractor(0, 80).cuda()
    nps_calculator = NPSCalculator(printfile, patch_size)
    nps_calculator = nps_calculator.cuda()
    total_variation = TotalVariation().cuda()

    imgsize = darknet_model.height
    image = Image.open(img_path).convert('RGB')
    label = np.loadtxt(lab_path)
    label = torch.from_numpy(label).float()
    if label.dim() == 1:
        label = label.unsqueeze(0)
    image, label = pad_and_scale(image, label, imgsize)
    transform = transforms.ToTensor()
    image = transform(image)

    batch_size = 24
    image = image.unsqueeze(0)
    label = label.unsqueeze(0)
    img_batch = image.expand(batch_size,-1,-1,-1)
    lab_batch = label.expand(batch_size,-1,-1)

    img_batch = img_batch.cuda()
    lab_batch = lab_batch.cuda()

    n_epochs = 10

    adv_patch_cpu = torch.rand((3,patch_size,patch_size))
    adv_patch_cpu.requires_grad_(True)

    optimizer = optim.Adam([adv_patch_cpu], lr = 0.01)

    tl1 = time.time()
    for epoch in range(n_epochs):
        ffw1 = time.time()
        adv_patch = adv_patch_cpu.cuda()
        tl0 = time.time()
        #print('batch load time: ', tl0-tl1)
        img_size = img_batch.size(-1)
        #print('transforming patches')
        t0 = time.time()
        adv_batch_t = patch_transformer(adv_patch, lab_batch, img_size)
        #print('applying patches')
        t1 = time.time()
        p_img_batch = patch_applier(img_batch, adv_batch_t)
        p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))
        #print('running patched images through model')
        t2 = time.time()

        output = darknet_model(p_img_batch)
        #print('does output require grad? ',output.requires_grad)
        #print('extracting max probs')
        t3 = time.time()
        max_prob = prob_extractor(output)
        #print('does max_prob require grad? ',max_prob.requires_grad)
        #print('calculating nps')
        t4 = time.time()
        nps = nps_calculator(adv_patch)
        t5 = time.time()
        #print('calculating tv')
        tv = total_variation(adv_patch)
        t6 = time.time()

        print('---------------------------------')
        print('   patch transformation : %f' % (t1-t0))
        print('      patch application : %f' % (t2-t1))
        print('        darknet forward : %f' % (t3-t2))
        print(' probability extraction : %f' % (t4-t3))
        print('        nps calculation : %f' % (t5-t4))
        print('        total variation : %f' % (t6-t5))
        print('---------------------------------')
        print('     total forward pass : %f' % (t6-t0))

        print(torch.mean(max_prob))
        print(nps)
        print(tv)


        det_loss = torch.mean(max_prob)
        nps_loss = nps*0.1
        tv_loss = tv*0.00005
        loss = det_loss + nps_loss + tv_loss

        img_batch.retain_grad()
        adv_batch_t.retain_grad()
        adv_patch.retain_grad()

        print('loss to patch', torch.autograd.grad(loss,img_batch))
        loss.backward()
        tl1 = time.time()
        print('adv_patch.grad',adv_patch_cpu.grad)
        print('adv_batch_t.grad',adv_batch_t.grad)
        print('img_batch.grad',img_batch.grad)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        ffw2 = time.time()
        if (epoch)%5 == 0:
            print('  EPOCH NR: ', epoch)
            print('EPOCH LOSS: ', loss.detach().cpu().numpy())
            print('  DET LOSS: ', det_loss.detach().cpu().numpy())
            print('  NPS LOSS: ', nps_loss.detach().cpu().numpy())
            print('   TV LOSS: ', tv_loss.detach().cpu().numpy())
            print('EPOCH TIME: ', ffw2-ffw1)
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.show()
        del adv_batch_t, output, max_prob


    from utils import *
    patch_size = 300
    img_size = darknet_model.height

    img_dir_v = "inria/Test/pos"
    lab_dir_v = "inria/Test/yolo-labels"

    adv_patch = Image.open("patch.png").convert('RGB')
    # adv_patch = Image.open("data/cmx.jpg").convert('RGB')
    transform = transforms.ToTensor()
    adv_patch = transform(adv_patch).cuda()


    train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir_v, lab_dir_v, 14, img_size, shuffle=True),
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=10)

    for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
        img_size = img_batch.size(-1)
        adv_batch_t = patch_transformer(adv_patch, lab_batch.cuda(), img_size, do_rotate=True, rand_loc=False)
        p_img = patch_applier(img_batch.cuda(), adv_batch_t)
        p_img = F.interpolate(p_img,(darknet_model.height, darknet_model.width))
        output = darknet_model(p_img)
        boxes = get_region_boxes(output,0.5,darknet_model.num_classes,
                             darknet_model.anchors, darknet_model.num_anchors)[0]
        boxes = nms(boxes,0.4)
        class_names = load_class_names('coco.names')
        squeezed = p_img.squeeze(0)
        print(squeezed.shape)
        img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
        plotted_image = plot_boxes(img, boxes, class_names=class_names)
        # plt.imshow(plotted_image)
        # plt.show()

    # apply an image as patch
    patch_size = adv_patch.size(-1)

    horse = Image.open("patch.png").convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    horse = tf(horse)
    transform = transforms.ToTensor()
    horse = transform(horse)

    adv_batch_t = patch_transformer(horse.cuda(), label.cuda(), img_size)
    p_img = patch_applier(image.cuda(), adv_batch_t)
    p_img = F.interpolate(p_img,(darknet_model.height, darknet_model.width))
    output = darknet_model(p_img)
    boxes = get_region_boxes(output,0.5,darknet_model.num_classes,
                             darknet_model.anchors, darknet_model.num_anchors)[0]
    boxes = nms(boxes,0.4)
    class_names = load_class_names('coco.names')
    squeezed = p_img.squeeze(0)
    im = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
    plotted_image = plot_boxes(im, boxes, class_names=class_names)
    plt.imshow(plotted_image)
    plt.show()




