import cv2 
import numpy as np

def get_ds(out_ly,W,H,frame,conf,threshold):
  bboxes = []
  conf_set = []
  clases_id = []
  class_list=[]
  for out in out_ly:
  
    for d in out:
      
      scores = d[5:]
      c_id = np.argmax(scores)
      confidence = scores[c_id]
   
      
      if confidence > conf:
        
        box = d[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")
   
        
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
   
       
        bboxes.append([x, y, int(width), int(height)])
        conf_set.append(float(confidence))
        clases_id.append(c_id)
        class_list.append(etiquetas[c_id])
        #print("Se ha detectado: ", etiquetas[c_id])
        idxs = cv2.dnn.NMSBoxes(bboxes, conf_set, conf,threshold)
        if len(idxs) > 0:
  
          for i in idxs.flatten():
            
            (x, y) = (bboxes[i][0], bboxes[i][1])
            (w, h) = (bboxes[i][2], bboxes[i][3])
               
            if clases_id[i]==0:
              cv2.rectangle(frame, (x, y), (x + w, y + h), colores_with, 2)
            else:
              cv2.rectangle(frame, (x, y), (x + w, y + h), colores_without, 2)
            
           
            text = "{}: {:.4f}".format(etiquetas[clases_id[i]], conf_set[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 2)
  return class_list, frame

ruta_clases="classes.names"
arquitectura="yolov3mod.cfg"
modelo="facemask.weights"
etiquetas = open(ruta_clases).read().strip().split("\n")
np.random.seed(42)
colores_with = (255,0,0)
colores_without = (0,0,255)
nn = cv2.dnn.readNetFromDarknet(arquitectura, modelo) 
l_names = nn.getLayerNames()

l_names = [l_names[i[0] - 1] for i in nn.getUnconnectedOutLayers()]
print(l_names)
conf=0.5
threshold=0.3

capture=cv2.VideoCapture('3.mp4')
frame_width = int(capture.get(3))

frame_height = int(capture.get(4))

print("frame width: ", frame_width)
print("frame height: ", frame_height)

out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20, (frame_width,frame_height))

print("processing. ")
total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
i=1
while True:
  ret,frame=capture.read()
  if ret==True:
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),  swapRB=True, crop=False)
    nn.setInput(blob)
    out_ly = nn.forward(l_names)
    c,f=get_ds(out_ly,frame_width,frame_height,frame,conf,threshold)
    out.write(f)
    print(i,"/",total)
    i+=1
  else:
    break
capture.release()
out.release()
print("done::")