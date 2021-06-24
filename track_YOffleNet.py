import sys
sys.path.insert(0, './YOffleNet')

import argparse
from sys import platform
import torch.backends.cudnn as cudnn

from YOffleNet.models.experimental import *
from YOffleNet.utils.datasets import *
from YOffleNet.utils.utils import *
from deep_sort import DeepSort
from torchinfo import summary
deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height, bbox_left, bbox_top, bbox_w, bbox_h):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def pointInPolygon(polyCorners, polyX, polyY, x, y):
    oddNodes = False
    j = polyCorners - 1

    for i in range (0, polyCorners, 1):
        if (polyY[i]< y and polyY[j]>=y or polyY[j]< y and polyY[i]>=y) and (polyX[i]<=x or polyX[j]<=x) :
            if polyX[i]+(y-polyY[i])/(polyY[j]-polyY[i])*(polyX[j]-polyX[i]) < x :
                if oddNodes == True:
                    oddNodes = False
                else :
                    oddNodes = True
        j = i
    return oddNodes


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    print("my device = ", device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # 2021.05.20 혜정 수정
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half = device.type != 'cpu' and torch.cuda.device_count() == 1  # half precision only supported on single-GPU

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    #summary(model, (1, 3, 640, 640))
    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
        print('We are in Webcam!===================')
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    print('time goes on!===================\n')
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    people_path = {}
    count = 0
    N = 0
    S = 0
    people_count = {}

    for path, img, im0s, vid_cap in dataset:
        print("my device = ", device)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        #t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        

        # Process detections
        # 하나의 frame에 여러 BBOX를 찾음
        for i, det in enumerate(pred):
            
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                print('We are in Path!===================')
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            points = np.array([[350,360], [1850, 700], [1810, 845], [150,470]], np.int32)  # outdoor
            #points = np.array([[380,300], [500,300], [500, 420], [380, 420]], np.int32)   # indoor
            cv2.polylines(im0, [points], True, (255,0,0), 2)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                
                # Write results 
                # 하나의 BBOX
                for *xyxy, conf, cls in det:

                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_bottom = max([xyxy[1].item(), xyxy[3].item()])
                    
                    if cls == 0 :

                        img_h, img_w, _ = im0.shape  # get image shape
                        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, bbox_left, bbox_top, bbox_w, bbox_h)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])
                        label = '%s %.2f' % (names[int(cls)], conf)
                        outputs = deepsort.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)) , im0)
                        
                        
                        # tracking 성공
                        if len(outputs) > 0 :
                            
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -1]
                            draw_boxes(im0, bbox_xyxy, identities)
                            

                            # People Counting & Direction
                            for i, box in enumerate(bbox_xyxy):
            
                                x1, y1, x2, y2 = [int (i) for i in box]
                                id = int(identities[i]) if identities is not None else 0

                                if pointInPolygon(4, [350, 1850, 1810, 150], [360, 700, 845, 470], x1, y2):
                                    if id not in people_count.keys() :
                                        count = count + 1
                                        people_count[id] = [x1]
                                    else:
                                        people_count[id].append(x1)
                                    cv2.circle(im0, (x1, y2), 5, (0,0,255), -1);
                                else:
                                    if id in people_count.keys() :
                                        first = people_count[id][0]
                                        last = people_count[id][-1]
                                        if first > last : S = S + 1
                                        else : N = N + 1
                                        del(people_count[id])
                            
            cv2.rectangle(im0, (10, 10), (750, 190), (255, 255, 255), -1)
            cv2.putText(im0, "People Counting = " + str(count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),3)
            cv2.putText(im0, "Northward = " + str(N), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0),3)
            cv2.putText(im0, "Southward = " + str(S), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0),3)

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (image with detections)

            # Stream results
            if True:
                cv2.namedWindow(p, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)

                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
    

    print('Done. (%.3fs)' % (time.time() - t0), end = ' ')
    print('(%.3f fps)' % (443/(time.time() - t0)))

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='YOffleNet/Weight/COCO/yolov4s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='YOffleNet/data/videos/test.mp4', help='source')  # file/folder, 0 for webcam 'inference/video'
    parser.add_argument('--output', type=str, default='YOffleNet_Out/yolov4s(cpu)', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
    
